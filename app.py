import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
import os

# Page config
st.set_page_config(
    page_title="Email Summarizer",
    page_icon="📧",
    layout="wide"
)

# Title and description
st.title("📧 AI Email Summarizer")
st.markdown("""
Transform lengthy emails into concise, actionable summaries using state-of-the-art Transformer models.
""")

# Sidebar - Model Selection
st.sidebar.header("⚙️ Settings")
model_choice = st.sidebar.radio(
    "Select Model:",
    ["Pretrained (IrisWiris/email-summarizer)", "Fine-tuned (Custom)"]
)

# Load model based on selection
@st.cache_resource
def load_model(model_type):
    if model_type == "Pretrained (IrisWiris/email-summarizer)":
        model_name = "IrisWiris/email-summarizer"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    else:
        # Load fine-tuned model from local directory
        model_path = "email.summa/email_model_finetuned"
        
        if not os.path.exists(model_path):
            st.error(f"⚠️ Fine-tuned model not found at '{model_path}'")
            st.info("Please complete training in your Jupyter notebook first, then restart this app.")
            st.stop()
        
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True)
    
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
    return summarizer

# Load the selected model
with st.spinner("Loading model..."):
    summarizer = load_model(model_choice)
    st.sidebar.success("✅ Model loaded!")

# Summary parameters
st.sidebar.subheader("Summary Parameters")
max_length = st.sidebar.slider("Max Summary Length", 50, 200, 100)
min_length = st.sidebar.slider("Min Summary Length", 20, 100, 30)

# Main content area
col1, col2 = st.columns(2)

with col1:
    st.subheader("📝 Input Email")
    email_text = st.text_area(
        "Paste your email here:",
        height=400,
        placeholder="Enter or paste a long email that you want to summarize...",
        key="email_input"
    )
    
    # Sample email button
    if st.button("📄 Load Sample Email"):
        st.session_state.email_input = """Subject: Q1 Project Status and Budget Review

Hi Team,

I wanted to give you a comprehensive update on our Q1 marketing project. We've completed the market research phase successfully, gathering data from over 500 respondents. The insights show strong potential in the 25-35 age demographic.

However, we're facing some budget constraints. The initial $50K allocation may need a 15% increase due to rising social media advertising costs. I've prepared a detailed cost breakdown for review.

The creative team has developed three campaign concepts, all tested with focus groups. Results indicate Concept B resonates best with our target audience.

Regarding timeline concerns from last week's meeting, we're slightly behind but can still meet the Q2 launch if we expedite the design phase and allocate additional resources.

Please review the attached documents before Friday's 2 PM meeting.

Best regards,
Sarah"""
        st.rerun()

with col2:
    st.subheader("✨ Generated Summary")
    
    if st.button("🚀 Generate Summary", type="primary"):
        if email_text:
            with st.spinner("Generating summary..."):
                try:
                    # Generate summary
                    summary = summarizer(
                        email_text,
                        max_length=max_length,
                        min_length=min_length,
                        do_sample=False
                    )
                    
                    # Display results
                    st.success("Summary generated successfully!")
                    st.markdown("### Summary:")
                    st.info(summary[0]['summary_text'])
                    
                    # Statistics
                    st.markdown("### 📊 Statistics")
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Original Words", len(email_text.split()))
                    with col_b:
                        st.metric("Summary Words", len(summary[0]['summary_text'].split()))
                    with col_c:
                        compression = (1 - len(summary[0]['summary_text'].split()) / len(email_text.split())) * 100
                        st.metric("Compression", f"{compression:.1f}%")
                    
                except Exception as e:
                    st.error(f"Error generating summary: {str(e)}")
        else:
            st.warning("Please enter an email to summarize!")

# Footer with explanation
st.markdown("---")
with st.expander("ℹ️ How it works"):
    st.markdown("""
    ### About the Transformer Model
    
    **How Transformers Work:**
    - Uses **self-attention mechanism** to understand relationships between words
    - Processes entire input sequences in parallel (unlike RNNs)
    - Encoder reads the email, Decoder generates the summary
    
    **Why Transformers for Summarization:**
    - Captures long-range dependencies in text
    - Understands context across entire email
    - Pre-trained on massive text corpora for general language understanding
    
    **Fine-tuning:**
    - Adapts pretrained weights to email-specific patterns
    - Updates mainly the attention layers and output head
    - Requires email-summary pairs for training
    
    **Limitations:**
    - May hallucinate facts not in original email
    - Can miss subtle nuances or important details
    - Performance depends on email style similarity to training data
    """)
