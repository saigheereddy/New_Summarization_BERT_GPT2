import streamlit as st

def main():

  st.title('Project - Text Summarisation')
  st.write("""
  #News Summarizer App
  This app gives a summarized information on *News Articles*
  """)
  input_val = "Hiatus"
  input_article = st.text_area("Article :",input_val,height=350)

  st.sidebar.header('User Input Features')
  models = ["T5-Transformer","BERT-Classifier"]
  choice = st.sidebar.selectbox('Model to use: ',models)
  if choice == "T5-Transformer":
    st.subheader("T5-Transformer Load here")
  elif choice == "BERT-Classifier":
    st.subheader("BERT-Classifier Load here")

if __name__ = "__main__":
  main()
