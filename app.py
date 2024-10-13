"""
Contract Extractor API

This API provides endpoints to extract contract details from PDF and DOCX files.
It uses a Hugging Face model to extract the details.

Endpoints:
- /extract: Extract contract details from multiple files
- /files: Get the list of uploaded files
- /chat/{contract_id}: Interact with a specific contract document using a chat interface
"""

import json
import requests
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from gradio_client import Client
import PyPDF2 
import os
from docx import Document

app = FastAPI()

# Initialize the Gradio client
client = Client("Jyothikamalesh/Vendor-contract-extractor")

# Create a directory to store the uploaded files
UPLOAD_FOLDER = "uploaded_files"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def extract_text_from_pdf(pdf_file):
    """
    Extract text from a PDF file.

    Args:
    pdf_file (str): Path to the PDF file

    Returns:
    str: Extracted text
    """
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(docx_file):
    """
    Extract text from a DOCX file.

    Args:
    docx_file (str): Path to the DOCX file

    Returns:
    str: Extracted text
    """
    doc = Document(docx_file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

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

        # Try to parse the response as JSON
        try:
            extracted_details = json.loads(response)
        except json.JSONDecodeError:
            # If parsing as JSON fails, try to extract relevant information from the response
            extracted_details = {}
            lines = response.splitlines()
            for line in lines:
                if ":" in line:
                    key, value = line.split(":", 1)
                    extracted_details[key.strip()] = value.strip()

        # Remove extra double quotes from keys
        extracted_details = remove_extra_quotes(extracted_details)

        return extracted_details
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}

def remove_extra_quotes(contract_details):
    new_contract_details = {}
    for key, value in contract_details.items():
        new_key = key.replace('"', '')
        if isinstance(value, str):
            new_contract_details[new_key] = value.replace('"', '')
        else:
            new_contract_details[new_key] = value
    return new_contract_details

@app.post("/extract")
async def extract_details_from_files(files: list[UploadFile] = File(...)):
    """
    Extract contract details from multiple files.

    Args:
    files (list[UploadFile]): List of files to extract details from

    Returns:
    dict: Extracted contract details
    """
    results = []
    for file in files:
        # Save the uploaded file
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(file_path, "wb") as saved_file:
            saved_file.write(file.file.read())

        # Extract text from the PDF or DOCX
        if file.filename.endswith(".pdf"):
            pdf_text = extract_text_from_pdf(file_path)
            if not pdf_text:
                raise HTTPException(status_code=400, detail="Failed to extract text from the PDF")
            extracted_details = extract_contract_details_with_model(pdf_text)

        elif file.filename.endswith(".docx"):
            docx_text = extract_text_from_docx(file_path)
            if not docx_text:
                raise HTTPException(status_code=400, detail="Failed to extract text from the DOCX file")
            extracted_details = extract_contract_details_with_model(docx_text)

        else:
            raise HTTPException(status_code=400, detail="Only PDF and DOCX files are allowed")

        results.append({
            "file_name": file.filename,
            "contract_details": extracted_details
        })

    # Return the extracted details in JSON format
    return JSONResponse(content={"results": results}, media_type="application/json")

def get_uploaded_files():
    """
    Get the list of uploaded files.

    Returns:
    list: List of uploaded files
    """
    return [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith((".pdf", ".docx"))]

@app.get("/files")
async def get_files():
    """
    Get the list of uploaded files.

    Returns:
    dict: List of uploaded files
    """
    uploaded_files = get_uploaded_files()
    return JSONResponse(content={"files": uploaded_files}, media_type="application/json")

def get_file_text(file_name):
    """
    Get the text of a specific file (PDF or DOCX).

    Args:
    file_name (str): Name of the file

    Returns:
    str: Extracted text
    """
    file_path = os.path.join(UPLOAD_FOLDER, file_name)
    if file_name.endswith(".pdf"):
        return extract_text_from_pdf(file_path)
    elif file_name.endswith(".docx"):
        return extract_text_from_docx(file_path)
    else:
        raise ValueError("Unsupported file type")

@app.post("/chat/{contract_id}")
async def chat_with_contract(contract_id: str, message: str):
    """
    Interact with a specific contract document using a chat interface.

    Args:
    contract_id (str): ID of the contract
    message (str): User message

    Returns:
    dict: Response from the model
    """
    uploaded_files = get_uploaded_files()
    if contract_id not in uploaded_files:
        raise HTTPException(status_code=404, detail="Contract not found")

    file_text = get_file_text(contract_id)

    # Prepare the prompt for the Gradio model
    prompt = f"""
            Extract the following details from the given contract and give them in JSON format:

            Vendor name, contract id, start date, end date, term of contract, next renewal year, scope, type of contract (multiple or single product), contract type (SAAS/Software/Fixed Bid/OEM), number of licenses in contract, cost per license, total license cost, renewal cost, maintenance cost, any other cost, any one-time cost or misc cost, total contract value, annual contract value, First Year P&L impact, Second Year P&L impact, Third Year P&L impact, Fourth Year P&L impact, Fifth Year P&L impact, First year Cash payments, Second year Cash payments, Third year Cash payments, Fourth year Cash payments, Fifth year Cash payments, change in scope with respect to years, change in scope in ﹩ terms, whether YoY change in scope is volume driven, YoY change in active months of contract, Increase in the cost of product/service as agreed to in the contract with vendor (CPI impact %), Increase in the cost of product/service as agreed to in the contract with vendor (CPI impact ﹩), If there is a change in rate/expense mentioned in the contract for next year.

            Contract text: {file_text}

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

        # Try to parse the response as JSON
        try:
            extracted_details = json.loads(response)
            # Format the extracted details
            formatted_details = {
                "results": [
                    {
                        "file_name": contract_id,
                        "contract_details": {}
                    }
                ]
            }
            for key, value in extracted_details.items():
                if value == "null":
                    formatted_details["results"][0]["contract_details"][key] = None
                elif value.isdigit():
                    formatted_details["results"][0]["contract_details"][key] = int(value)
                else:
                    formatted_details["results"][0]["contract_details"][key] = value
            return formatted_details
        except json.JSONDecodeError:
            # If parsing as JSON fails, try to extract relevant information from the response
            extracted_details = {}
            lines = response.splitlines()
            for line in lines:
                if ":" in line:
                    key, value = line.split(":", 1)
                    if value.strip() == "null":
                        extracted_details[key.strip()] = None
                    elif value.strip().isdigit():
                        extracted_details[key.strip()] = int(value.strip())
                    else:
                        extracted_details[key.strip()] = value.strip()
            formatted_details = {
                "results": [
                    {
                        "file_name": contract_id,
                        "contract_details": extracted_details
                    }
                ]
            }
            return formatted_details
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)