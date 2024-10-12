import json
import requests
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from gradio_client import Client
import PyPDF2 
import os

app = FastAPI()

client = Client("Jyothikamalesh/Vendor-contract-extractor")

# Create a directory to store the uploaded PDF files
UPLOAD_FOLDER = "uploaded_pdfs"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Helper function to interact with Hugging Face model for extracting contract details
def extract_contract_details_with_model(text):
    prompt = f"""
            Extract the following details from the given contract and give them in JSON format:

            Vendor name, contract id, start date, end date, term of contract, next renewal year, scope, type of contract (multiple or single product), contract type (SAAS/Software/Fixed Bid/OEM), number of licenses in contract, cost per license, total license cost, renewal cost, maintenance cost, any other cost, any one-time cost or misc cost, total contract value, annual contract value, First Year P&L impact, Second Year P&L impact, Third Year P&L impact, Fourth Year P&L impact, Fifth Year P&L impact, First year Cash payments, Second year Cash payments, Third year Cash payments, Fourth year Cash payments, Fifth year Cash payments, change in scope with respect to years, change in scope in ﹩ terms, whether YoY change in scope is volume driven, YoY change in active months of contract, Increase in the cost of product/service as agreed to in the contract with vendor (CPI impact %), Increase in the cost of product/service as agreed to in the contract with vendor (CPI impact ﹩), If there is a change in rate/expense mentioned in the contract for next year.

            Contract text: {text}
            """

    system_message = ""
    max_tokens = 512
    temperature = 0.7
    top_p = 0.95
    
    try:
        # Call the Gradio model with the prepared prompt
        response = client.predict(
            message=prompt,
            system_message=system_message,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            api_name="/chat"
        )

        # Check if the response is already in JSON format
        if response.startswith("{") and response.endswith("}"):
            extracted_details = json.loads(response)
        else:
            # If not, try to parse the response as JSON
            try:
                extracted_details = json.loads(response)
            except json.JSONDecodeError:
                return {"error": "Failed to parse model output as JSON"}

        return extracted_details
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}

# Endpoint to upload multiple contracts and extract details
@app.post("/extract")
async def extract_details_from_pdfs(files: list[UploadFile] = File(...)):
    if not all(file.filename.endswith(".pdf") for file in files):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    results = []
    for file in files:
        # Save the uploaded PDF file
        pdf_file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(pdf_file_path, "wb") as pdf_file:
            pdf_file.write(file.file.read())

        # Extract text from the PDF
        pdf_text = extract_text_from_pdf(pdf_file_path)
        if not pdf_text:
            raise HTTPException(status_code=400, detail="Failed to extract text from the PDF")

        # Pass the extracted text to the Gradio model for information extraction
        extracted_details = extract_contract_details_with_model(pdf_text)

        results.append({
            "file_name": file.filename,
            "contract_details": extracted_details
        })

    # Return the extracted details in JSON format
    return JSONResponse(content={"results": results}, media_type="application/json")

# Function to get the list of uploaded PDF files
def get_uploaded_pdfs():
    pdf_files = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith(".pdf")]
    return pdf_files

# Function to get the text of a specific PDF file
def get_pdf_text(file_name):
    pdf_file_path = os.path.join(UPLOAD_FOLDER, file_name)
    with open(pdf_file_path, "rb") as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text

# Endpoint to get the list of uploaded PDF files
@app.get("/pdfs")
async def get_pdfs():
    pdf_files = get_uploaded_pdfs()
    return JSONResponse(content={"pdfs": pdf_files}, media_type="application/json")

# Endpoint to interact with a specific contract document using a chat interface
# Endpoint to interact with a specific contract document using a chat interface
@app.post("/chat/{contract_id}")
async def chat_with_contract(contract_id: str, message: str):
    pdf_files = get_uploaded_pdfs()
    if contract_id not in pdf_files:
        raise HTTPException(status_code=404, detail="Contract not found")

    pdf_text = get_pdf_text(contract_id)

    # Prepare the prompt for the Gradio model
    prompt = f"""
            Extract the following details from the given contract and give them in JSON format:

            Vendor name, contract id, start date, end date, term of contract, next renewal year, scope, type of contract (multiple or single product), contract type (SAAS/Software/Fixed Bid/OEM), number of licenses in contract, cost per license, total license cost, renewal cost, maintenance cost, any other cost, any one-time cost or misc cost, total contract value, annual contract value, First Year P&L impact, Second Year P&L impact, Third Year P&L impact, Fourth Year P&L impact, Fifth Year P&L impact, First year Cash payments, Second year Cash payments, Third year Cash payments, Fourth year Cash payments, Fifth year Cash payments, change in scope with respect to years, change in scope in ﹩ terms, whether YoY change in scope is volume driven, YoY change in active months of contract, Increase in the cost of product/service as agreed to in the contract with vendor (CPI impact %), Increase in the cost of product/service as agreed to in the contract with vendor (CPI impact ﹩), If there is a change in rate/expense mentioned in the contract for next year.

            Contract text: {pdf_text}

            User: {message}
            """

    system_message = ""
    max_tokens = 512
    temperature = 0.7
    top_p = 0.95

    try:
        # Call the Gradio model with the prepared prompt
        response = client.predict(
            message=prompt,
            system_message=system_message,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            api_name="/chat"
        )

        print("Model output:")
        print(response)

        # Try to parse the response as JSON
        try:
            extracted_details = json.loads(response)
            return extracted_details
        except json.JSONDecodeError:
            # If parsing as JSON fails, return the response as plain text
            return {"response": response}
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)