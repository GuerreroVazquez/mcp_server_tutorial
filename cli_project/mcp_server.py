from mcp.server.fastmcp import FastMCP
from pydantic import Field
from mcp.server.fastmcp.prompts import base
# creating the MDK with the python SDK
mcp = FastMCP("DocumentMCP", log_level="ERROR")


docs = {
    "deposition.md": "This deposition covers the testimony of Angela Smith, P.E.",
    "report.pdf": "The report details the state of a 20m condenser tower.",
    "financials.docx": "These financials outline the project's budget and expenditures.",
    "outlook.pdf": "This document presents the projected future performance of the system.",
    "plan.md": "The plan outlines the steps for the project's implementation.",
    "spec.txt": "These specifications define the technical requirements for the equipment.",
}

# TODO: Write a tool to read a doc

@mcp.tool("read_doc", description="Reads the contents of a document given its ID.")
def read_doc(doc_id: str = Field(description="The ID of the document to read.")) -> str:
    # this will generate the json schema for us.
    if doc_id not in docs:
        raise ValueError(f"Document with ID '{doc_id}' not found.")
    return docs[doc_id]

# TODO: Write a tool to edit a doc
@mcp.tool("edit_doc", description="Edits the contents of a document given its ID and new content.")
def edit_doc(doc_id: str = Field(description="The ID of the document to edit."), 
             old_text: str = Field(description="The old content for the document. Must match perfectly."),
             new_text: str = Field(description="The new content for the document.")) -> str:
    if doc_id not in docs:
        raise ValueError(f"Document with ID '{doc_id}' not found.")
    if old_text not in docs[doc_id]:
        raise ValueError(f"The old text provided does not match the current content of the document.")
    docs[doc_id] = docs[doc_id].replace(old_text, new_text)
    return docs[doc_id]


@mcp.resource("docs://documents", mime_type="application/json", description="A collection of documents.")
def list_docs() -> list[str]:
    return list(docs.keys())

@mcp.resource("docs://documents/{doc_id}", mime_type="text/plain", description="The content of a specific document.")
def fetch_doc(doc_id: str) -> str:
    if doc_id not in docs:
        raise ValueError(f"Document with ID '{doc_id}' not found.")
    return docs[doc_id]

@mcp.prompt(name="format", description="Formats a document's content based on the specified style.")
def format_document(
    doc_id: str = Field(description="The ID of the document to format.")
) -> list[base.Message]:
    prompt = f"""
    Your goal is to format the content of the document with ID '{doc_id}' to markdown style. The content of the document is as follows:
    <content>
    {docs[doc_id]}
    </content>
    Please provide the formatted content in markdown style.
    """
    return [base.UserMessage(content=prompt)]
    

# TODO: Write a prompt to summarize a doc


if __name__ == "__main__":
    mcp.run(transport="stdio")
