import pdfplumber, sys
sys.stdout.reconfigure(encoding='utf-8')
with pdfplumber.open(r'C:\Users\Carla Batista\Downloads\Desafio-bootcamp-CDIA.docx (1).pdf') as pdf:
    print(f'Total pages: {len(pdf.pages)}')
    for i, page in enumerate(pdf.pages):
        print(f'--- PAGE {i+1} ---')
        text = page.extract_text()
        if text:
            print(text)
        print()
