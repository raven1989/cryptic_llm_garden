import os

def main():
    svg_path = "/Users/louie/AppleRepo/cryptic_llm_garden/wiki/media/moe_evolution_timeline.svg"

    lines = []
    lines.append('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 520" width="960" height="520">')
    lines.append('  <style>')
    lines.append('    text {')
    lines.append("      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;")
    lines.append('    }')
    lines.append('  </style>')

    lines.append('  <defs>')
    lines.append('    <marker id="arrow-claude" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto">')
    lines.append('      <polygon points="0 2, 8 5, 0 8" fill="#4a4a4a"/>')
    lines.append('    </marker>')
    lines.append('    <filter id="shadow-soft" x="-10%" y="-10%" width="120%" height="120%">')
    lines.append('      <feDropShadow dx="0" dy="2" stdDeviation="4" flood-color="#000000" flood-opacity="0.06"/>')
    lines.append('    </filter>')
    lines.append('  </defs>')

    # Warm cream background
    lines.append('  <rect width="960" height="520" fill="#f8f6f3"/>')

    # Title & Subtitle
    lines.append('  <text x="480" y="45" text-anchor="middle" fill="#1a1a1a" font-size="20" font-weight="800" letter-spacing="0.03em">CHRONOLOGICAL EVOLUTION OF MIXTURE OF EXPERTS (MoE)</text>')
    lines.append('  <text x="480" y="70" text-anchor="middle" fill="#6a6a6a" font-size="13" font-weight="400">Chronological breakthroughs in sparse conditional computation from early neural networks to DeepSeek-V3</text>')

    # Node coordinates
    # Row 1 (y = 170): Col 1, Col 2, Col 3, Col 4
    # Row 2 (y = 390): Col 4, Col 3, Col 2 (Col 1 is the legend)

    nodes = [
        # Milestone 1: Jacobs et al.
        {
            "id": "m1", "x": 120, "y": 170, "color": "#e8e6e3",
            "title": "1991 • Jacobs et al.", "subtitle": "Adaptive Local Experts",
            "bullets": ["• Supervised ensemble", "• Gating network router"]
        },
        # Milestone 2: Shazeer et al.
        {
            "id": "m2", "x": 360, "y": 170, "color": "#a8c5e6",
            "title": "2017 • Shazeer et al.", "subtitle": "Sparsely-Gated MoE",
            "bullets": ["• Noisy Top-K gating", "• Auxiliary load-balance loss"]
        },
        # Milestone 3: GShard
        {
            "id": "m3", "x": 600, "y": 170, "color": "#9dd4c7",
            "title": "2020 • GShard", "subtitle": "MoE in Transformers",
            "bullets": ["• Top-2 random routing", "• Static Expert Capacity"]
        },
        # Milestone 4: Switch Transformer
        {
            "id": "m4", "x": 840, "y": 170, "color": "#9dd4c7",
            "title": "2021 • Switch", "subtitle": "Switch Transformer",
            "bullets": ["• Simplified Top-1 routing", "• Scale-invariant balance loss"]
        },
        # Milestone 5: ST-MoE
        {
            "id": "m5", "x": 840, "y": 390, "color": "#f4e4c1",
            "title": "2022 • ST-MoE", "subtitle": "Stability &amp; Scale",
            "bullets": ["• Router Z-Loss prevents inf", "• Expert Dropout regularizer"]
        },
        # Milestone 6: DeepSeek-V2
        {
            "id": "m6", "x": 600, "y": 390, "color": "#f4e4c1",
            "title": "2024 • DeepSeek-V2", "subtitle": "Shared Expert Isolation",
            "bullets": ["• Routed &amp; shared expert split", "• Latent KV cache (MLA)"]
        },
        # Milestone 7: DeepSeek-V3
        {
            "id": "m7", "x": 360, "y": 390, "color": "#f4e4c1",
            "title": "2026 • DeepSeek-V3", "subtitle": "Aux-Loss-Free Routing",
            "bullets": ["• Dynamic routing bias", "• Sequence load balancing"]
        }
    ]

    # Draw Nodes
    for n in nodes:
        x_c, y_c = n["x"], n["y"]
        # Box bounds: x-90 to x+90, y-55 to y+55
        x, y = x_c - 90, y_c - 55
        lines.append(f'  <!-- Node {n["id"]}: {n["title"]} -->')
        lines.append(f'  <rect x="{x}" y="{y}" width="180" height="110" rx="12" ry="12" fill="{n["color"]}" stroke="#4a4a4a" stroke-width="2.2" filter="url(#shadow-soft)"/>')

        # Content
        lines.append(f'  <text x="{x_c}" y="{y_c - 31}" text-anchor="middle" font-size="12.5" font-weight="700" fill="#1a1a1a">{n["title"]}</text>')
        lines.append(f'  <text x="{x_c}" y="{y_c - 13}" text-anchor="middle" font-size="11.5" font-weight="600" fill="#4a4a4a">{n["subtitle"]}</text>')
        lines.append(f'  <text x="{x_c - 76}" y="{y_c + 14}" text-anchor="start" font-size="10.5" fill="#5a5a5a">{n["bullets"][0]}</text>')
        lines.append(f'  <text x="{x_c - 76}" y="{y_c + 32}" text-anchor="start" font-size="10.5" fill="#5a5a5a">{n["bullets"][1]}</text>')

    # Draw Legend Box (Col 1, Row 2: x=120, y=390)
    legend_x_c, legend_y_c = 120, 390
    lx, ly = legend_x_c - 90, legend_y_c - 55
    lines.append('  <!-- Legend Box -->')
    lines.append(f'  <rect x="{lx}" y="{ly}" width="180" height="110" rx="12" ry="12" fill="#ffffff" stroke="#4a4a4a" stroke-width="1.8" stroke-dasharray="4,4"/>')
    lines.append(f'  <text x="{legend_x_c}" y="{legend_y_c - 33}" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1a1a" letter-spacing="0.05em">ERA KEY</text>')

    # Legend Items
    items = [
        ("#e8e6e3", "Early Foundations (1991)"),
        ("#a8c5e6", "Top-K Sparsity (2017)"),
        ("#9dd4c7", "Transformer Scale"),
        ("#f4e4c1", "Stability &amp; SOTA")
    ]

    for idx, (color, text) in enumerate(items):
        item_y = legend_y_c - 15 + idx * 17
        lines.append(f'  <circle cx="{legend_x_c - 72}" cy="{item_y}" r="4.5" fill="{color}" stroke="#4a4a4a" stroke-width="1.2"/>')
        lines.append(f'  <text x="{legend_x_c - 60}" y="{item_y + 3.5}" text-anchor="start" font-size="10" font-weight="500" fill="#5a5a5a">{text}</text>')

    # Draw Arrows & Labels
    arrows = [
        # M1 -> M2
        {"x1": 210, "y1": 170, "x2": 270, "y2": 170, "label": "Noisy Top-K", "lx": 240, "ly": 156, "align": "middle"},
        # M2 -> M3
        {"x1": 450, "y1": 170, "x2": 510, "y2": 170, "label": "Scaling FFNs", "lx": 480, "ly": 156, "align": "middle"},
        # M3 -> M4
        {"x1": 690, "y1": 170, "x2": 750, "y2": 170, "label": "Capacity Logic", "lx": 720, "ly": 156, "align": "middle"},
        # M4 -> M5 (Down)
        {"x1": 840, "y1": 225, "x2": 840, "y2": 335, "label": "bf16 stability", "lx": 855, "ly": 284, "align": "start"},
        # M5 -> M6 (Left)
        {"x1": 750, "y1": 390, "x2": 690, "y2": 390, "label": "Specialization", "lx": 720, "ly": 376, "align": "middle"},
        # M6 -> M7 (Left)
        {"x1": 510, "y1": 390, "x2": 450, "y2": 390, "label": "Aux-Loss-Free", "lx": 480, "ly": 376, "align": "middle"}
    ]

    for a in arrows:
        lines.append(f'  <!-- Connection: {a["label"]} -->')
        lines.append(f'  <line x1="{a["x1"]}" y1="{a["y1"]}" x2="{a["x2"]}" y2="{a["y2"]}" stroke="#4a4a4a" stroke-width="2.2" marker-end="url(#arrow-claude)"/>')

        # Draw background badge for arrow label to prevent overlap
        if a["align"] == "middle":
            # For horizontal lines, draw a nice label badge
            text_len = len(a["label"]) * 6
            lines.append(f'  <rect x="{a["lx"] - text_len/2 - 4}" y="{a["ly"] - 9}" width="{text_len + 8}" height="13" fill="#f8f6f3" rx="3" opacity="0.95"/>')
            lines.append(f'  <text x="{a["lx"]}" y="{a["ly"]}" text-anchor="middle" font-size="10" font-weight="700" fill="#4a4a4a">{a["label"]}</text>')
        else:
            # For vertical lines, offset label slightly
            text_len = len(a["label"]) * 6
            lines.append(f'  <rect x="{a["lx"] - 2}" y="{a["ly"] - 9}" width="{text_len + 4}" height="13" fill="#f8f6f3" rx="3" opacity="0.95"/>')
            lines.append(f'  <text x="{a["lx"]}" y="{a["ly"]}" text-anchor="start" font-size="10" font-weight="700" fill="#4a4a4a">{a["label"]}</text>')

    # Footer
    lines.append('  <text x="480" y="495" text-anchor="middle" fill="#9ca3af" font-size="10" font-weight="500">Wiki Companion Series: MoE Chronological Milestones. Generated 2026.</text>')

    lines.append('</svg>')

    os.makedirs(os.path.dirname(svg_path), exist_ok=True)
    with open(svg_path, "w") as f:
        f.write("\n".join(lines))
    print(f"SVG generated successfully at {svg_path}")

if __name__ == "__main__":
    main()
