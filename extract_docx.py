from docx import Document
# Function to read .docx and extract text
def load_docx(path):
    doc = Document(path)
    text = "\n".join([p.text for p in doc.paragraphs if p.text.strip() != ""])
    return text
# -----------------------
# MAIN
# -----------------------
if __name__ == "__main__":
    # Change this to your resume file path
    resume_path = "Nitin Nanje Gowda Resume.docx"
    job_path = "Job Description.docx"

    # Extract text
    resume_text = load_docx(resume_path)
    job_text = load_docx(job_path)

    # Print extracted text
    print("\n📄 Resume Text:\n")
    print(resume_text)

    print("\n📄 Job Description Text:\n")
    print(job_text)
