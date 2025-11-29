import os
import json
import logging
import markdown
import pdfkit

from ai_steward.tools.corrector import DraftNode

def save_secure_report(refined_draft: DraftNode, output_base_path: str):
    output_dir = os.path.dirname(output_base_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Created directory: {output_dir}")

    md_path = f"{output_base_path}.md"
    try:
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(refined_draft.draft)
        logging.info(f"Saved Draft (Markdown): {md_path}")
    except Exception as e:
        logging.error(f"Failed to save Markdown: {e}", exc_info=True)

    json_path = f"{output_base_path}_meta.json"
    try:
        meta_data = {
            "citations": getattr(refined_draft, "citations", []),
            "escalation_suggestions": getattr(refined_draft, "escalation_suggestions", []),
            "thinkings": getattr(refined_draft, "thinkings", {})
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, ensure_ascii=False, indent=2)
        logging.info(f"Saved Metadata (JSON): {json_path}")
    except Exception as e:
        logging.error(f"Failed to save Metadata: {e}", exc_info=True)

    pdf_path = f"{output_base_path}.pdf"
    
    css_style = """
    <style>
        @page { margin: 20mm; size: A4; }
        body {
            font-family: "Hiragino Mincho ProN", "Yu Mincho", "MS PMincho", "Noto Serif CJK JP", serif;
            line-height: 1.6;
            color: #2c3e50;
            font-size: 10.5pt;
            word-wrap: break-word;
            overflow-wrap: break-word;
        }
        a { color: #2980b9; text-decoration: none; border-bottom: 1px dotted #2980b9; }
        a:hover { text-decoration: underline; }
        h1 {
            font-family: "Hiragino Kaku Gothic ProN", "Yu Gothic", "Meiryo", sans-serif;
            font-size: 24pt; border-bottom: 3px solid #34495e; 
            padding-bottom: 10px; margin-bottom: 30px; margin-top: 50px;
        }
        h2 {
            font-family: "Hiragino Kaku Gothic ProN", "Yu Gothic", "Meiryo", sans-serif;
            font-size: 16pt; border-left: 6px solid #e74c3c; 
            padding-left: 12px; margin-top: 40px; margin-bottom: 20px;
            background-color: #f9f9f9; padding-top: 5px; padding-bottom: 5px;
            page-break-after: avoid;
        }
        h3 {
            font-family: "Hiragino Kaku Gothic ProN", "Yu Gothic", "Meiryo", sans-serif;
            font-size: 12pt; font-weight: bold; margin-top: 25px;
            border-bottom: 1px dashed #bdc3c7;
        }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; font-size: 9pt; }
        th { background-color: #34495e; color: white; padding: 10px; border: 1px solid #34495e; }
        td { border: 1px solid #bdc3c7; padding: 8px; vertical-align: top; }
        tr:nth-child(even) { background-color: #f2f2f2; }
        pre {
            background-color: #f8f8f8; padding: 12px; border: 1px solid #ddd;
            border-radius: 4px; overflow-x: hidden;
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        code {
            font-family: "Consolas", "Monaco", "Courier New", monospace;
            font-size: 9pt; color: #c7254e;
        }
        blockquote { border-left: 4px solid #ddd; padding-left: 15px; color: #7f8c8d; }
        h2#restricted-access-authorization-requests {
            color: #c0392b; border-left-color: #c0392b;
            page-break-before: always;
        }
    </style>
    """

    try:
        html_content = markdown.markdown(
            refined_draft.draft, 
            extensions=['tables', 'fenced_code', 'toc']
        )
        
        full_html = f"""
        <!DOCTYPE html>
        <html>
        <head><meta charset="UTF-8">{css_style}</head>
        <body>{html_content}</body>
        </html>
        """

        options = {
            'encoding': "UTF-8",
            'no-outline': None,
            'enable-local-file-access': None,
            'margin-top': '20mm', 'margin-right': '20mm',
            'margin-bottom': '20mm', 'margin-left': '20mm',
        }

        pdfkit.from_string(full_html, pdf_path, options=options)
        logging.info(f"Saved Report (PDF): {pdf_path}")
        
    except Exception as e:
        logging.error(f"PDF Conversion Failed: {e}")
        html_debug_path = f"{output_base_path}_fallback.html"
        try:
            with open(html_debug_path, "w", encoding="utf-8") as f:
                f.write(full_html)
            logging.warning(f"Saved fallback HTML to: {html_debug_path}")
        except Exception as html_e:
            logging.error(f"Failed to save fallback HTML: {html_e}")
