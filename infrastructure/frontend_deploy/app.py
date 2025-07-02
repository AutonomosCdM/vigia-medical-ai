import gradio as gr
import os

# Read the HTML content
def get_html_content():
    with open("hero-agent-smith.html", "r", encoding="utf-8") as f:
        return f.read()

# Custom CSS to ensure full viewport
custom_css = """
body, #root {
    margin: 0 !important;
    padding: 0 !important;
    height: 100vh !important;
    overflow: hidden !important;
}

.gradio-container {
    margin: 0 !important;
    padding: 0 !important;
    max-width: 100% !important;
    height: 100vh !important;
}

iframe {
    width: 100vw !important;
    height: 100vh !important;
    border: none !important;
    margin: 0 !important;
    padding: 0 !important;
}
"""

# Create Gradio interface
with gr.Blocks(
    title="Autonomos AiLab - Agent Smith Landing",
    css=custom_css,
    theme=gr.themes.Base()
) as demo:
    
    html_content = get_html_content()
    
    gr.HTML(
        value=html_content,
        elem_id="landing_page"
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True
    )