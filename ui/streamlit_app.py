import requests
import streamlit as st

st.title("DocuQuery")
api = st.sidebar.text_input("API URL", "http://127.0.0.1:8000")

with st.sidebar:
    pdf = st.file_uploader("Upload a PDF", type="pdf")
    if pdf and st.button("Index document"):
        r = requests.post(
            f"{api}/documents",
            files={"file": (pdf.name, pdf.getvalue(), "application/pdf")},
            timeout=300,
        )
        if r.ok:
            st.success(f"Indexed {r.json()['chunks']} chunks")
        else:
            st.error(r.text)

question = st.text_input("Ask a question about your documents")
if question:
    r = requests.post(f"{api}/ask", json={"question": question}, timeout=120)
    if r.ok:
        data = r.json()
        st.write(data["answer"])
        st.caption(f"{data['latency_ms']} ms")
        for s in data["sources"]:
            with st.expander(f"[{s['id']}] {s['filename']}, page {s['page']} (score {s['score']})"):
                st.write(s["snippet"])
    else:
        st.error(r.text)