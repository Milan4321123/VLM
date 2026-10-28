"""Render the editable project report to PDF. Requires requirements-report.txt."""
from html import escape
from pathlib import Path
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, PageBreak, Preformatted, Table, TableStyle, Spacer

ROOT = Path(__file__).resolve().parents[1]


def inline_markup(text):
    """Render the report's inline code without interpreting arbitrary HTML."""
    parts = re.split(r'(`[^`]+`|\*\*[^*]+\*\*|(?<!\*)\*[^*]+\*(?!\*))', text)
    rendered = []
    for part in parts:
        if part.startswith('`') and part.endswith('`'):
            rendered.append('<font name="Courier" size="9.2">' + escape(part[1:-1]) + '</font>')
        elif part.startswith('**') and part.endswith('**'):
            rendered.append('<b>' + escape(part[2:-2]) + '</b>')
        elif part.startswith('*') and part.endswith('*'):
            rendered.append('<i>' + escape(part[1:-1]) + '</i>')
        else:
            rendered.append(escape(part))
    return ''.join(rendered)


def page_furniture(canvas, doc):
    width, height = A4
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor('#CBD5E1'))
    canvas.line(52, height - 39, width - 52, height - 39)
    canvas.setFont('Helvetica', 8)
    canvas.setFillColor(colors.HexColor('#526174'))
    canvas.drawString(52, height - 29, 'VLM-FO1 / Qwen3.5-0.8B  |  Project report')
    canvas.drawString(52, 30, 'SoSe 2026  |  Inference-only scope')
    canvas.drawRightString(width - 52, 30, str(doc.page))
    canvas.restoreState()


def build():
    output = ROOT / 'output/pdf/project_report.pdf'
    output.parent.mkdir(parents=True, exist_ok=True)
    styles = {
        'title': ParagraphStyle('title', fontName='Helvetica-Bold', fontSize=21, leading=25,
                                textColor=colors.HexColor('#133C55'), spaceAfter=13, keepWithNext=True),
        'heading': ParagraphStyle('heading', fontName='Helvetica-Bold', fontSize=13, leading=17,
                                  textColor=colors.HexColor('#133C55'), spaceBefore=15, spaceAfter=8,
                                  keepWithNext=True),
        'subheading': ParagraphStyle('subheading', fontName='Helvetica-Bold', fontSize=11.5, leading=15,
                                     textColor=colors.HexColor('#133C55'), spaceBefore=12, spaceAfter=7,
                                     keepWithNext=True),
        'minorheading': ParagraphStyle('minorheading', fontName='Helvetica-Bold', fontSize=11, leading=15,
                                       textColor=colors.HexColor('#133C55'), spaceBefore=10, spaceAfter=7,
                                       keepWithNext=True),
        'quote': ParagraphStyle('quote', fontName='Helvetica-Oblique', fontSize=11, leading=15.3,
                                leftIndent=15, rightIndent=15, spaceAfter=10),
        'body': ParagraphStyle('body', fontName='Helvetica', fontSize=11, leading=15.3,
                               alignment=TA_LEFT, spaceAfter=10, allowWidows=0, allowOrphans=0),
        'reference': ParagraphStyle('reference', fontName='Helvetica', fontSize=10, leading=14,
                                   spaceAfter=12, allowWidows=0, allowOrphans=0),
        'code': ParagraphStyle('code', fontName='Courier', fontSize=8.6, leading=11.8,
                              backColor=colors.HexColor('#F1F5F9'), borderPadding=8,
                              spaceBefore=5, spaceAfter=13),
        'cell': ParagraphStyle('cell', fontName='Helvetica', fontSize=9.5, leading=13),
    }
    text = (ROOT / 'docs/project_report.md').read_text(encoding='utf-8')
    story = []
    # The paragraph scanner below deliberately treats headings, tables and code
    # as separate block types rather than flattening Markdown into body text.
    blocks = []
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        if not lines[index].strip():
            index += 1
            continue
        if lines[index].startswith('```'):
            index += 1
            code = []
            while index < len(lines) and not lines[index].startswith('```'):
                code.append(lines[index])
                index += 1
            if index == len(lines):
                raise ValueError('Unclosed code fence in report')
            blocks.append(('code', '\n'.join(code)))
            index += 1
        else:
            paragraph = []
            while index < len(lines) and lines[index].strip() and not lines[index].startswith('```'):
                paragraph.append(lines[index])
                index += 1
            blocks.append(('text', '\n'.join(paragraph)))
    for kind, block in blocks:
        if kind == 'code':
            story.append(Preformatted(block, styles['code'], maxLineLength=89))
            continue
        if block.startswith('|'):
            rows = []
            for line in block.splitlines():
                cells = [cell.strip() for cell in line.strip().strip('|').split('|')]
                if all(re.fullmatch(r':?-+:?', cell.replace(' ', '')) for cell in cells):
                    continue
                rows.append([Paragraph(inline_markup(cell), styles['cell']) for cell in cells])
            if any(len(row) != 2 for row in rows):
                raise ValueError('Report tables must have two columns')
            table = Table(rows, colWidths=[147, A4[0] - 104 - 147], repeatRows=1, hAlign='LEFT', splitByRow=0)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E2EAF1')),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('GRID', (0, 0), (-1, -1), .4, colors.HexColor('#CBD5E1')),
                ('LEFTPADDING', (0, 0), (-1, -1), 7),
                ('RIGHTPADDING', (0, 0), (-1, -1), 7),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ]))
            story.extend([table, Spacer(1, 12)])
            continue
        style = 'body'
        if block.startswith('# '):
            block, style = block[2:], 'title'
        elif block.startswith('#### '):
            block, style = block[5:], 'minorheading'
        elif block.startswith('### '):
            block, style = block[4:], 'subheading'
        elif block.startswith('## '):
            block, style = block[3:], 'heading'
            if block == 'References':
                story.append(PageBreak())
        elif re.match(r'^(?:\*\*)?\[\d+\]', block):
            style = 'reference'
        elif block.startswith('> '):
            block, style = block[2:], 'quote'
        if style == 'reference' or all(line.startswith('**') and line.endswith('**') for line in block.splitlines()):
            markup = '<br/>'.join(inline_markup(line) for line in block.splitlines())
        else:
            markup = inline_markup(block.replace('\n', ' '))
        story.append(Paragraph(markup, styles[style]))
    doc = SimpleDocTemplate(str(output), pagesize=A4, rightMargin=52, leftMargin=52,
                            topMargin=57, bottomMargin=51, title='Adapting VLM-FO1 to Qwen3.5',
                            author='Project submission', subject='Inference-only migration and verification')
    doc.build(story, onFirstPage=page_furniture, onLaterPages=page_furniture)
    print(output)


if __name__ == '__main__':
    build()
