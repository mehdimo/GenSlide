import traceback
import streamlit as st

from slide_gen import generate_presentation

def create_ui():
    st.write("""
# Gen Slides
### Generating powerpoint slides for your text
""")

    content = st.text_area(label="Enter your text:", height=400)
    try:
        if content:
            filename = generate_presentation(content)
            st.write(f"file {filename} is generated.")
    except Exception:
        st.write("Error in generating slides.")
        st.write(traceback.format_exc())

if __name__ == "__main__":
    create_ui()
