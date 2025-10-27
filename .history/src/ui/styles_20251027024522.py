"""
CSS styles for the Streamlit application.
"""

CUSTOM_CSS = """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(120deg, #2563eb, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        color: #64748b;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    .stChatMessage {
        background-color: #f8fafc;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .upload-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
    }
    
    .feature-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
</style>
"""


WELCOME_HTML = """
<div style='text-align: center; padding: 3rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white;'>
    <h2>👋 Welcome!</h2>
    <p style='font-size: 1.2rem; margin-top: 1rem;'>
        Upload a PDF document from the sidebar to start asking questions
    </p>
    <p style='margin-top: 1rem; opacity: 0.9;'>
        🚀 Powered by local AI models - Fast, Private, and Free!
    </p>
</div>
"""


def get_feature_cards_html() -> str:
    """Generate HTML for feature cards."""
    return """
    <div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 2rem; margin-top: 2rem;'>
        <div class='feature-card'>
            <h3>🔒 Private</h3>
            <p>All processing happens locally on your machine. Your documents never leave your computer.</p>
        </div>
        <div class='feature-card'>
            <h3>⚡ Fast</h3>
            <p>Optimized for quick responses using efficient embedding models and local LLMs.</p>
        </div>
        <div class='feature-card'>
            <h3>🎯 Accurate</h3>
            <p>Uses advanced semantic search to find the most relevant information in your documents.</p>
        </div>
    </div>
    """
