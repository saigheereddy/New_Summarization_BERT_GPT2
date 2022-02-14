import streamlit as st
import pickle
import t5transformer as t5
import bert_utils as bert
# import bert_transformer as bert
import meta as m
# PAGE_CONFIG = {"page_title":"StColab.io","page_icon":":smiley:","layout":"centered"}
# st.beta_set_page_config(**PAGE_CONFIG)
def computations_here(text_data,model_type):
	"""
	Do your computations here
	"""
	x = t5.test_df.loc[t5.test_df['text']==text_data]
	if model_type == "T5-Transformer":
		model_summary = t5.summarizeText(text_data)
	else:
		mod = bert.load_model(pretrained=True, path=f"./models/temp_encdec_weights.pth")
		model_summary = bert.summarise_articles(text_data, mod)
  
	return x['summary'].values,model_summary

def main():
	model = bert.load_model(pretrained=True, path=f"./models/temp_encdec_weights.pth")
	st.title("News Summarization")
	# st.subheader("Running StreamLit from CoLab")
	menu = ["Home","Preprocessing","Demo_Article","Custom_Article","Final Notes"]
	choice = st.sidebar.selectbox("Menu",menu)
	if choice == 'Home':
		st.image('/content/gdrive/Shareddrives/Text_Summarization_Project/images/Text-Summarization-front.jpg')
		st.markdown(m.SUMMARY_STORY,unsafe_allow_html=True)
		# st.markdown()
	elif choice == 'Preprocessing':
		t5.final_container
		# pass

	elif choice == 'Demo_Article':
		st.header(" Input News Article ")
		input_val = "Replace this text with News Article"
		# txt = st.text_area("Article :",placeholder=input_val,height=350)
		txt = st.selectbox('Test Article to Summarize : ',options=t5.test_df['text'])
		model = ['T5-Transformer','BERT-Classifier']
		model_choice = st.selectbox('Model ',model)
		# st.selectbox('Model to use:',model)
		st.header(" Output Summary ")
		if st.button("Summarize Article"):
			if model_choice == "T5-Transformer":
				original_summary, predicted_summary = computations_here(txt,model_choice)
			elif model_choice == "BERT-Classifier":
				predicted_summary="Summarizing with BERT-Classifier"
				original_summary = "Summarizing with BERT-Classifier"
		else:
			predicted_summary = "Generated Summary of the Article"
			original_summary = "Generated Summary of the Article"
		st.text_area("Predicted Summary :",predicted_summary,height=250)
		st.text_area("Original Summary :",original_summary,height=250)

	elif choice == 'Custom_Article':
		st.header(" Input News Article ")
		input_val = "Replace this text with News Article"
		txt = st.text_area("Article :",placeholder=input_val,height=350)
		model = ['T5-Transformer','BERT-Transformer']
		model_choice = st.selectbox('Model ',model)
		# st.selectbox('Model to use:',model)
		st.header(" Output Summary ")
		
		# Add model details here
		if st.button("Summarize Article"):
			if model_choice == "T5-Transformer":
				summary_val= t5.summarizeText(txt)
			elif model_choice == "BERT-Transformer":
				summary_val=bert.summarise_articles(txt, model)
		else:
			summary_val = "Generated Summary of the Article"
		st.text_area("Summary :",summary_val,height=250)
	elif choice == 'Final Notes':
		pass
if __name__ == '__main__':
	main()
