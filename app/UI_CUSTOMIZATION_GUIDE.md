# 🎨 MatSAM UI Customization Guide

This document provides clear instructions for modifying the MatSAM interface, branding, and technical references. All primary UI changes are made in `app/app.py`.

---

## 🏗️ 1. Core Branding & Page Setup
**Location:** `app/app.py` → `st.set_page_config()` (~Line 29)

To change the browser tab title or the emoji icon:
*   **Change Title:** Update `page_title="MatSAM - SEM Segmentation"`
*   **Change Icon:** Update `page_icon="🔬"`

---

## 🎨 2. Visual Styles (Colors & Fonts)
**Location:** `app/app.py` → `st.markdown(""" <style> ... </style> """)` (~Lines 36-64)

The app uses **Vanilla CSS** for its premium look.
*   **Main Gradient:** Find `background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);` inside `.main-header` and `.stButton>button`. Replace the HEX codes (e.g., `#667eea`) with your brand colors.
*   **Typography:** To change fonts, add `font-family: 'Inter', sans-serif;` inside the `.stButton` or `.main-header` classes.
*   **Metric Cards:** Adjust the `.metric-card` class to change the appearance of the result boxes (IoU, Grain Count).

---

## 📚 3. Adding Research References (MatSAM, Medium, etc.)
**Location:** `app/app.py` → `with st.expander("📖 About MatSAM & Technical Methodology"):` (~Line 635)

To add your papers:
1.  Scroll to the bottom of `app.py`.
2.  Add a new section using Markdown.
3.  **Example Code:**
    ```python
    st.markdown("### 📄 Publications")
    st.markdown("- [MATSAM: Efficient Microstructure Extraction](link_to_paper)")
    st.markdown("- [Medium: Multi-scale Pre-processing for SEM](link_to_blog)")
    ```

---

## 🤖 4. AI Consultant Customization
**Location:** `app/app.py` → `with tab3:` (~Line 590)

*   **Change Welcome Message:** Update the `st.markdown` text right under the tab definition.
*   **Default Prompt:** Change the text in `st.chat_input("How should I optimize...")` to your most common question.
*   **System Knowledge:** To change the "personality" or specific expertise of the AI, edit `app/utils/ai_consultant.py` → `_get_system_prompt()`.

---

## 🧪 5. Adjusting Lab Sliders
**Location:** `app/app.py` → `with enhance_col1:` (~Line 470)

*   **Change Ranges:** Each `st.slider` has arguments: `label, min_value, max_value, default_value, step`. 
    *   *Example:* To make `Global Blending` go up to 2.0, change `st.slider(..., 0.0, 1.5, ...)` to `0.0, 2.0`.
*   **Help Tooltips:** Modify the `help="..."` string to explain the parameter in your own words.

---

## 📂 6. File Locations Summary
| Purpose | File Path |
| :--- | :--- |
| Main UI, Tabs, Sliders, CSS | `app/app.py` |
| AI Logic, RAG System, Prompts | `app/utils/ai_consultant.py` |
| PDF Knowledge Base | `app/knowledge_base/` |
| Image Processing Logic | `app/utils/matsam_wrapper.py` |
| Model Initialization | `app/app.py` (`load_model_ui`) |

---

## 💡 Best Practices
1.  **Backup**: Before making major CSS changes, keep a copy of the original `st.markdown` style block.
2.  **Rerun**: After saving the file, click "Always Rerun" in the Streamlit browser window to see changes instantly.
3.  **Icons**: Use any [Streamlit-supported Emoji](https://share.streamlit.io/streamlit/emoji-shortcodes) for icons.
