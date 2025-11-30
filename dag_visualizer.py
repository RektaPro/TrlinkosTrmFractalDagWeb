"""
DAG Visualizer for T-RLINKOS

This module provides visualization tools for the FractalMerkleDAG
reasoning trace. It enables:

- Interactive HTML visualization (D3.js force-directed graph)
- GraphML export (for Gephi, yEd)
- DOT export (for Graphviz)
- JSON export (for custom integrations)
- Text-based explanation of reasoning paths

Usage:
    from dag_visualizer import DAGVisualizer
    from t_rlinkos_trm_fractal_dag import TRLinkosTRM

    # Run reasoning
    model = TRLinkosTRM(x_dim=64, y_dim=32, z_dim=64)
    y_pred, dag = model.forward_recursive(x_batch, max_steps=10)

    # Visualize
    visualizer = DAGVisualizer(dag)
    visualizer.to_html("reasoning_dag.html")
    visualizer.to_graphml("dag.graphml")
    print(visualizer.explain_path())
"""

import json
from typing import Dict, List, Optional, Any

from t_rlinkos_trm_fractal_dag import FractalMerkleDAG, DAGNode


class DAGVisualizer:
    """Génère des visualisations interactives du raisonnement T-RLINKOS.

    Formats supportés:
    - HTML interactif (D3.js force-directed graph)
    - GraphML (pour Gephi, yEd)
    - DOT (pour Graphviz)
    - JSON (pour intégration custom)

    Example:
        >>> from t_rlinkos_trm_fractal_dag import TRLinkosTRM, FractalMerkleDAG
        >>> import numpy as np
        >>> model = TRLinkosTRM(x_dim=32, y_dim=16, z_dim=32)
        >>> y_pred, dag = model.forward_recursive(np.random.randn(2, 32), max_steps=5)
        >>> viz = DAGVisualizer(dag)
        >>> viz.to_html("output.html")
        >>> print(viz.explain_path())
    """

    def __init__(self, dag: FractalMerkleDAG):
        """Initialise le visualiseur.

        Args:
            dag: FractalMerkleDAG contenant la trace de raisonnement
        """
        self.dag = dag

    def to_html(self, output_path: str = "dag_visualization.html") -> str:
        """Génère une visualisation HTML interactive avec D3.js.

        Crée un fichier HTML autonome avec une visualisation force-directed
        du DAG de raisonnement. Les nœuds sont interactifs (draggable, clickable).

        Args:
            output_path: Chemin du fichier HTML de sortie

        Returns:
            Chemin du fichier créé
        """
        nodes_data = []
        edges_data = []

        for node_id, node in self.dag.nodes.items():
            nodes_data.append({
                "id": node_id[:8],  # Tronquer pour lisibilité
                "full_id": node_id,
                "step": node.step,
                "depth": node.depth,
                "score": node.score if node.score is not None else 0,
                "is_best": node_id == self.dag.best_node_id,
                "y_hash": node.y_hash[:8],
                "z_hash": node.z_hash[:8],
            })

            for parent_id in node.parents:
                if parent_id in self.dag.nodes:
                    edges_data.append({
                        "source": parent_id[:8],
                        "target": node_id[:8],
                    })

        # Generate safe JSON
        nodes_json = json.dumps(nodes_data)
        edges_json = json.dumps(edges_data)

        # Precompute values for the template
        total_nodes = len(self.dag.nodes)
        best_score_str = f"{self.dag.best_score:.4f}" if self.dag.best_score > float('-inf') else 'N/A'

        html_template = f'''<!DOCTYPE html>
<html>
<head>
    <title>T-RLINKOS Reasoning DAG</title>
    <meta charset="UTF-8">
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #eee;
            min-height: 100vh;
        }}
        h1 {{
            color: #00d4ff;
            text-align: center;
            margin-bottom: 10px;
        }}
        .subtitle {{
            text-align: center;
            color: #888;
            margin-bottom: 20px;
        }}
        .container {{
            display: flex;
            gap: 20px;
        }}
        svg {{
            background: rgba(0, 0, 0, 0.3);
            border-radius: 10px;
            flex: 1;
        }}
        .node {{
            cursor: pointer;
        }}
        .node circle {{
            fill: #4a90a4;
            stroke: #00d4ff;
            stroke-width: 2px;
            transition: all 0.3s ease;
        }}
        .node circle:hover {{
            fill: #00d4ff;
            stroke: #fff;
        }}
        .node.best circle {{
            fill: #ff6b6b;
            stroke: #ff9999;
            stroke-width: 3px;
        }}
        .node.branch circle {{
            fill: #9b59b6;
            stroke: #d4a5ff;
        }}
        .node text {{
            font-size: 10px;
            fill: #fff;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.8);
        }}
        .link {{
            fill: none;
            stroke: #00d4ff;
            stroke-opacity: 0.4;
            stroke-width: 2px;
        }}
        #info {{
            width: 300px;
            background: rgba(0, 0, 0, 0.5);
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #00d4ff;
        }}
        #info h3 {{
            color: #00d4ff;
            margin-top: 0;
            border-bottom: 1px solid #00d4ff;
            padding-bottom: 10px;
        }}
        .info-row {{
            margin: 8px 0;
            display: flex;
            justify-content: space-between;
        }}
        .info-label {{
            color: #888;
        }}
        .info-value {{
            color: #fff;
            font-family: monospace;
        }}
        .best-badge {{
            background: #ff6b6b;
            color: white;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 12px;
        }}
        .stats {{
            text-align: center;
            margin: 10px 0;
            color: #888;
        }}
        .legend {{
            display: flex;
            justify-content: center;
            gap: 20px;
            margin: 10px 0;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 5px;
        }}
        .legend-circle {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
        }}
    </style>
</head>
<body>
    <h1>🧠 T-RLINKOS Reasoning DAG</h1>
    <p class="subtitle">Interactive visualization of recursive reasoning trace</p>
    <div class="legend">
        <div class="legend-item">
            <div class="legend-circle" style="background: #4a90a4; border: 2px solid #00d4ff;"></div>
            <span>Standard Node</span>
        </div>
        <div class="legend-item">
            <div class="legend-circle" style="background: #ff6b6b; border: 2px solid #ff9999;"></div>
            <span>Best Node</span>
        </div>
        <div class="legend-item">
            <div class="legend-circle" style="background: #9b59b6; border: 2px solid #d4a5ff;"></div>
            <span>Branch Node</span>
        </div>
    </div>
    <p class="stats">Total nodes: {total_nodes} | Best score: {best_score_str}</p>
    <div class="container">
        <svg width="900" height="600"></svg>
        <div id="info">
            <h3>📍 Node Information</h3>
            <p style="color: #888;">Click on a node to see details</p>
        </div>
    </div>
    <script>
        const nodes = {nodes_json};
        const links = {edges_json};

        const svg = d3.select("svg");
        const width = 900, height = 600;

        // Create arrow marker for directed edges
        svg.append("defs").append("marker")
            .attr("id", "arrowhead")
            .attr("viewBox", "-0 -5 10 10")
            .attr("refX", 20)
            .attr("refY", 0)
            .attr("orient", "auto")
            .attr("markerWidth", 6)
            .attr("markerHeight", 6)
            .append("path")
            .attr("d", "M 0,-5 L 10,0 L 0,5")
            .attr("fill", "#00d4ff")
            .style("opacity", 0.6);

        const simulation = d3.forceSimulation(nodes)
            .force("link", d3.forceLink(links).id(d => d.id).distance(80))
            .force("charge", d3.forceManyBody().strength(-400))
            .force("center", d3.forceCenter(width / 2, height / 2))
            .force("y", d3.forceY().y(d => 50 + d.step * 80).strength(0.3));

        const link = svg.append("g")
            .selectAll("line")
            .data(links)
            .join("line")
            .attr("class", "link")
            .attr("marker-end", "url(#arrowhead)");

        const node = svg.append("g")
            .selectAll("g")
            .data(nodes)
            .join("g")
            .attr("class", d => {{
                let cls = "node";
                if (d.is_best) cls += " best";
                if (d.depth > 0) cls += " branch";
                return cls;
            }})
            .call(d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended));

        node.append("circle")
            .attr("r", d => 8 + Math.min(Math.abs(d.score) * 3, 12));

        node.append("text")
            .attr("dy", -15)
            .attr("text-anchor", "middle")
            .text(d => `S${{d.step}}${{d.depth > 0 ? ' (D' + d.depth + ')' : ''}}`);

        node.on("click", (event, d) => {{
            const scoreStr = d.score !== 0 ? d.score.toFixed(4) : 'N/A';
            document.getElementById("info").innerHTML = `
                <h3>📍 Node: ${{d.id}}</h3>
                ${{d.is_best ? '<span class="best-badge">⭐ BEST NODE</span>' : ''}}
                <div class="info-row">
                    <span class="info-label">Step:</span>
                    <span class="info-value">${{d.step}}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Depth:</span>
                    <span class="info-value">${{d.depth}} (fractal level)</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Score:</span>
                    <span class="info-value">${{scoreStr}}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">y_hash:</span>
                    <span class="info-value">${{d.y_hash}}...</span>
                </div>
                <div class="info-row">
                    <span class="info-label">z_hash:</span>
                    <span class="info-value">${{d.z_hash}}...</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Full ID:</span>
                    <span class="info-value" style="font-size: 9px;">${{d.full_id.substring(0, 32)}}...</span>
                </div>
            `;
        }});

        simulation.on("tick", () => {{
            link.attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);
            node.attr("transform", d => `translate(${{d.x}},${{d.y}})`);
        }});

        function dragstarted(event) {{
            if (!event.active) simulation.alphaTarget(0.3).restart();
            event.subject.fx = event.subject.x;
            event.subject.fy = event.subject.y;
        }}

        function dragged(event) {{
            event.subject.fx = event.x;
            event.subject.fy = event.y;
        }}

        function dragended(event) {{
            if (!event.active) simulation.alphaTarget(0);
            event.subject.fx = null;
            event.subject.fy = null;
        }}
    </script>
</body>
</html>
'''
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_template)

        return output_path

    def to_graphml(self, output_path: str = "dag.graphml") -> str:
        """Export vers GraphML pour Gephi/yEd.

        Crée un fichier GraphML compatible avec les outils de visualisation
        de graphes comme Gephi et yEd.

        Args:
            output_path: Chemin du fichier GraphML de sortie

        Returns:
            Chemin du fichier créé
        """
        graphml = '''<?xml version="1.0" encoding="UTF-8"?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns
         http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">
  <key id="step" for="node" attr.name="step" attr.type="int"/>
  <key id="depth" for="node" attr.name="depth" attr.type="int"/>
  <key id="score" for="node" attr.name="score" attr.type="double"/>
  <key id="is_best" for="node" attr.name="is_best" attr.type="boolean"/>
  <key id="y_hash" for="node" attr.name="y_hash" attr.type="string"/>
  <key id="z_hash" for="node" attr.name="z_hash" attr.type="string"/>
  <graph id="ReasoningDAG" edgedefault="directed">
'''
        for node_id, node in self.dag.nodes.items():
            is_best = "true" if node_id == self.dag.best_node_id else "false"
            score = node.score if node.score is not None else 0.0
            graphml += f'''    <node id="{node_id[:16]}">
      <data key="step">{node.step}</data>
      <data key="depth">{node.depth}</data>
      <data key="score">{score}</data>
      <data key="is_best">{is_best}</data>
      <data key="y_hash">{node.y_hash[:16]}</data>
      <data key="z_hash">{node.z_hash[:16]}</data>
    </node>
'''
        edge_id = 0
        for node_id, node in self.dag.nodes.items():
            for parent_id in node.parents:
                if parent_id in self.dag.nodes:
                    graphml += f'    <edge id="e{edge_id}" source="{parent_id[:16]}" target="{node_id[:16]}"/>\n'
                    edge_id += 1

        graphml += '''  </graph>
</graphml>'''

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(graphml)

        return output_path

    def to_dot(self, output_path: str = "dag.dot") -> str:
        """Export vers DOT pour Graphviz.

        Crée un fichier DOT qui peut être rendu avec Graphviz (dot, neato, etc.).

        Args:
            output_path: Chemin du fichier DOT de sortie

        Returns:
            Chemin du fichier créé

        Example:
            >>> viz.to_dot("dag.dot")
            >>> # Then run: dot -Tpng dag.dot -o dag.png
        """
        dot = "digraph ReasoningDAG {\n"
        dot += "  rankdir=TB;\n"
        dot += "  bgcolor=\"#1a1a2e\";\n"
        dot += "  node [shape=circle, style=filled, fontcolor=white];\n"
        dot += "  edge [color=\"#00d4ff\"];\n\n"

        # Group nodes by step for ranking
        steps: Dict[int, List[str]] = {}
        for node_id, node in self.dag.nodes.items():
            if node.step not in steps:
                steps[node.step] = []
            steps[node.step].append(node_id)

        for node_id, node in self.dag.nodes.items():
            if node_id == self.dag.best_node_id:
                color = "#ff6b6b"
                penwidth = "3"
            elif node.depth > 0:
                color = "#9b59b6"
                penwidth = "2"
            else:
                color = "#4a90a4"
                penwidth = "2"

            score_str = f"{node.score:.3f}" if node.score is not None else "N/A"
            label = f"S{node.step}\\nD{node.depth}\\n{score_str}"
            dot += f'  "{node_id[:8]}" [label="{label}", fillcolor="{color}", penwidth="{penwidth}"];\n'

        dot += "\n"

        for node_id, node in self.dag.nodes.items():
            for parent_id in node.parents:
                if parent_id in self.dag.nodes:
                    dot += f'  "{parent_id[:8]}" -> "{node_id[:8]}";\n'

        # Add rank constraints for layout
        dot += "\n"
        for step, node_ids in sorted(steps.items()):
            if len(node_ids) > 1:
                ids_str = " ".join(f'"{nid[:8]}"' for nid in node_ids)
                dot += f"  {{ rank=same; {ids_str} }}\n"

        dot += "}\n"

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(dot)

        return output_path

    def to_json(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Export vers JSON pour intégration custom.

        Args:
            output_path: Chemin optionnel du fichier JSON de sortie.
                         Si None, retourne seulement le dictionnaire.

        Returns:
            Dictionnaire avec la structure du DAG
        """
        data = {
            "metadata": {
                "total_nodes": len(self.dag.nodes),
                "best_node_id": self.dag.best_node_id,
                "best_score": self.dag.best_score if self.dag.best_score > float('-inf') else None,
                "depth_stats": self.dag.get_depth_statistics(),
                "root_nodes": self.dag.root_nodes,
            },
            "nodes": [],
            "edges": [],
        }

        for node_id, node in self.dag.nodes.items():
            data["nodes"].append({
                "id": node_id,
                "step": node.step,
                "depth": node.depth,
                "score": node.score,
                "y_hash": node.y_hash,
                "z_hash": node.z_hash,
                "parents": node.parents,
                "children": node.children,
                "is_best": node_id == self.dag.best_node_id,
                "branch_root": node.branch_root,
            })

            for parent_id in node.parents:
                data["edges"].append({
                    "source": parent_id,
                    "target": node_id,
                })

        if output_path is not None:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)

        return data

    def explain_path(self, node_id: Optional[str] = None) -> str:
        """Génère une explication textuelle du chemin de raisonnement.

        Crée une description lisible du chemin de raisonnement du nœud
        spécifié jusqu'à la racine, montrant chaque étape avec ses
        métadonnées.

        Args:
            node_id: ID du nœud à expliquer. Si None, utilise le meilleur nœud.

        Returns:
            Explication textuelle formatée du chemin de raisonnement
        """
        if node_id is None:
            node_id = self.dag.best_node_id

        if node_id is None:
            return "No reasoning path available."

        path = self.dag.get_fractal_path(node_id)

        explanation = "\n🧠 T-RLINKOS REASONING TRACE\n"
        explanation += "=" * 50 + "\n\n"

        for i, node in enumerate(path):
            is_best = " ⭐ BEST NODE" if node.node_id == self.dag.best_node_id else ""
            is_branch = " 🌿 BRANCH" if node.depth > 0 else ""

            score_str = f"{node.score:.4f}" if node.score is not None else "N/A"

            explanation += f"📍 STEP {i}: {node.node_id[:8]}...{is_best}{is_branch}\n"
            explanation += f"   ├── Step: {node.step}\n"
            explanation += f"   ├── Depth: {node.depth} (fractal level)\n"
            explanation += f"   ├── Score: {score_str}\n"
            explanation += f"   ├── y_hash: {node.y_hash[:16]}...\n"
            explanation += f"   └── z_hash: {node.z_hash[:16]}...\n"
            explanation += "\n"

        explanation += "=" * 50 + "\n"
        explanation += f"📊 Summary:\n"
        explanation += f"   • Total steps in path: {len(path)}\n"
        explanation += f"   • Total nodes in DAG: {len(self.dag.nodes)}\n"
        best_score_str = f"{self.dag.best_score:.4f}" if self.dag.best_score > float('-inf') else "N/A"
        explanation += f"   • Best score: {best_score_str}\n"

        depth_stats = self.dag.get_depth_statistics()
        explanation += f"   • Depth distribution: {depth_stats}\n"

        return explanation

    def get_summary(self) -> Dict[str, Any]:
        """Retourne un résumé des statistiques du DAG.

        Returns:
            Dictionnaire avec les statistiques clés du DAG
        """
        return {
            "total_nodes": len(self.dag.nodes),
            "root_nodes_count": len(self.dag.root_nodes),
            "best_node_id": self.dag.best_node_id[:16] if self.dag.best_node_id else None,
            "best_score": self.dag.best_score if self.dag.best_score > float('-inf') else None,
            "depth_stats": self.dag.get_depth_statistics(),
            "max_depth": max(self.dag.get_depth_statistics().keys()) if self.dag.nodes else 0,
        }


# ============================
#  Tests
# ============================


if __name__ == "__main__":
    import numpy as np
    from t_rlinkos_trm_fractal_dag import TRLinkosTRM, FractalMerkleDAG

    print("=" * 60)
    print("DAG Visualizer - Tests")
    print("=" * 60)

    np.random.seed(42)

    # --- Test 1: Basic visualization ---
    print("\n--- Test 1: Basic visualization ---")

    # Create a simple model and run reasoning
    model = TRLinkosTRM(x_dim=32, y_dim=16, z_dim=32, hidden_dim=64, num_experts=2)
    x_batch = np.random.randn(2, 32)
    target = np.random.randn(2, 16)

    def scorer(x, y):
        return -np.mean((y - target) ** 2, axis=-1)

    y_pred, dag = model.forward_recursive(
        x_batch, max_steps=5, scorer=scorer, backtrack=True
    )

    print(f"[Test 1] DAG nodes: {len(dag.nodes)}")
    print(f"[Test 1] Best score: {dag.best_score:.4f}")

    # Create visualizer
    viz = DAGVisualizer(dag)

    # Test to_html
    html_path = "/tmp/test_dag.html"
    result = viz.to_html(html_path)
    print(f"[Test 1] HTML exported to: {result}")
    assert result == html_path

    # Verify file was created
    with open(html_path, 'r') as f:
        content = f.read()
    assert "T-RLINKOS" in content
    assert "d3.js" in content.lower() or "d3.v7" in content
    print("[Test 1] ✅ HTML export works correctly!")

    # --- Test 2: GraphML export ---
    print("\n--- Test 2: GraphML export ---")

    graphml_path = "/tmp/test_dag.graphml"
    result = viz.to_graphml(graphml_path)
    print(f"[Test 2] GraphML exported to: {result}")

    with open(graphml_path, 'r') as f:
        content = f.read()
    assert "graphml" in content
    assert "<node id=" in content
    assert "<edge id=" in content
    print("[Test 2] ✅ GraphML export works correctly!")

    # --- Test 3: DOT export ---
    print("\n--- Test 3: DOT export ---")

    dot_path = "/tmp/test_dag.dot"
    result = viz.to_dot(dot_path)
    print(f"[Test 3] DOT exported to: {result}")

    with open(dot_path, 'r') as f:
        content = f.read()
    assert "digraph" in content
    assert "->" in content  # Edge notation
    print("[Test 3] ✅ DOT export works correctly!")

    # --- Test 4: JSON export ---
    print("\n--- Test 4: JSON export ---")

    json_path = "/tmp/test_dag.json"
    data = viz.to_json(json_path)

    assert "metadata" in data
    assert "nodes" in data
    assert "edges" in data
    assert data["metadata"]["total_nodes"] == len(dag.nodes)
    print(f"[Test 4] JSON nodes: {len(data['nodes'])}")
    print(f"[Test 4] JSON edges: {len(data['edges'])}")

    # Verify file was created
    import os
    assert os.path.exists(json_path)
    print("[Test 4] ✅ JSON export works correctly!")

    # --- Test 5: explain_path ---
    print("\n--- Test 5: explain_path ---")

    explanation = viz.explain_path()
    print(f"[Test 5] Explanation length: {len(explanation)} chars")
    assert "T-RLINKOS REASONING TRACE" in explanation
    assert "STEP" in explanation
    assert "Score" in explanation
    print("[Test 5] Sample output:")
    print(explanation[:500] + "..." if len(explanation) > 500 else explanation)
    print("[Test 5] ✅ explain_path works correctly!")

    # --- Test 6: get_summary ---
    print("\n--- Test 6: get_summary ---")

    summary = viz.get_summary()
    print(f"[Test 6] Summary: {summary}")
    assert "total_nodes" in summary
    assert "best_score" in summary
    assert "depth_stats" in summary
    print("[Test 6] ✅ get_summary works correctly!")

    # --- Test 7: Fractal DAG visualization ---
    print("\n--- Test 7: Fractal DAG visualization ---")

    # Test with fractal branching
    y_pred_fractal, dag_fractal = model.forward_recursive_fractal(
        x_batch,
        max_steps=5,
        scorer=scorer,
        fractal_branching=True,
        branch_threshold=0.01,
    )

    viz_fractal = DAGVisualizer(dag_fractal)
    depth_stats = dag_fractal.get_depth_statistics()
    print(f"[Test 7] Fractal DAG nodes: {len(dag_fractal.nodes)}")
    print(f"[Test 7] Depth stats: {depth_stats}")

    html_fractal_path = "/tmp/test_dag_fractal.html"
    viz_fractal.to_html(html_fractal_path)
    print(f"[Test 7] Fractal HTML exported to: {html_fractal_path}")
    print("[Test 7] ✅ Fractal DAG visualization works correctly!")

    # --- Test 8: Empty DAG handling ---
    print("\n--- Test 8: Empty DAG handling ---")

    empty_dag = FractalMerkleDAG()
    empty_viz = DAGVisualizer(empty_dag)

    empty_explanation = empty_viz.explain_path()
    assert "No reasoning path available" in empty_explanation
    print("[Test 8] Empty DAG explanation handled correctly")

    empty_summary = empty_viz.get_summary()
    assert empty_summary["total_nodes"] == 0
    print("[Test 8] ✅ Empty DAG handling works correctly!")

    print("\n" + "=" * 60)
    print("✅ All DAG Visualizer tests passed!")
    print("=" * 60)
