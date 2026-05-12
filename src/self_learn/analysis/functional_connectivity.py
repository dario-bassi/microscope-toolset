"""Functional connectivity analysis from calcium imaging traces.

Builds directed functional connectivity graphs from multi-neuron calcium
imaging data. Supports both spontaneous activity correlation and
stimulation-evoked cascade analysis.

Functions:
    correlation_matrix      -- Pairwise Pearson correlation between traces
    lagged_correlation      -- Time-shifted cross-correlation (directionality)
    build_adjacency         -- Threshold correlations into binary adjacency
    cascade_connectivity    -- Infer directed graph from stimulation cascades
    graph_metrics           -- Degree, centrality, clustering, hub detection
"""

import numpy as np


def correlation_matrix(traces, method='pearson'):
    """Compute pairwise correlation matrix between calcium traces.

    Args:
        traces: 2D array (n_neurons, n_frames) or dict of {id: 1D trace}.
        method: 'pearson' (default) or 'spearman'.

    Returns:
        dict with:
            matrix: 2D array (n, n) of correlation values [-1, 1].
            labels: list of neuron IDs/indices.
            mean_correlation: float, mean of upper triangle.
    """
    if isinstance(traces, dict):
        labels = sorted(traces.keys())
        mat = np.array([traces[k] for k in labels], dtype=np.float64)
    else:
        mat = np.asarray(traces, dtype=np.float64)
        labels = list(range(mat.shape[0]))

    n = mat.shape[0]
    corr = np.zeros((n, n))

    if method == 'spearman':
        # Rank-transform each trace
        ranked = np.zeros_like(mat)
        for i in range(n):
            ranked[i] = np.argsort(np.argsort(mat[i])).astype(float)
        mat = ranked

    # Compute Pearson on (possibly rank-transformed) data
    for i in range(n):
        for j in range(i, n):
            xi = mat[i] - mat[i].mean()
            xj = mat[j] - mat[j].mean()
            denom = np.sqrt(np.sum(xi**2) * np.sum(xj**2))
            if denom > 1e-10:
                corr[i, j] = float(np.sum(xi * xj) / denom)
            else:
                corr[i, j] = 0.0
            corr[j, i] = corr[i, j]

    # Mean of upper triangle
    mask = np.triu_indices(n, k=1)
    mean_corr = float(np.mean(corr[mask])) if n > 1 else 0.0

    return {
        'matrix': corr,
        'labels': labels,
        'mean_correlation': round(mean_corr, 4),
    }


def lagged_correlation(traces, max_lag=10):
    """Compute time-lagged cross-correlation to detect directionality.

    For each pair (i, j), finds the lag at which trace_j best correlates
    with trace_i. Positive optimal lag means i leads j (i→j connection).

    Args:
        traces: 2D array (n_neurons, n_frames) or dict.
        max_lag: Maximum lag (frames) to test in each direction.

    Returns:
        dict with:
            optimal_lag: 2D int array (n, n). lag[i,j] > 0 means i leads j.
            peak_correlation: 2D float array (n, n). Best correlation at optimal lag.
            labels: list of neuron IDs.
            directed_pairs: list of (source, target, lag, correlation) for
                significant directed relationships.
    """
    if isinstance(traces, dict):
        labels = sorted(traces.keys())
        mat = np.array([traces[k] for k in labels], dtype=np.float64)
    else:
        mat = np.asarray(traces, dtype=np.float64)
        labels = list(range(mat.shape[0]))

    n, T = mat.shape
    opt_lag = np.zeros((n, n), dtype=int)
    peak_corr = np.zeros((n, n))

    for i in range(n):
        xi = mat[i] - mat[i].mean()
        xi_norm = np.sqrt(np.sum(xi**2))
        if xi_norm < 1e-10:
            continue

        for j in range(n):
            if i == j:
                continue
            xj = mat[j] - mat[j].mean()
            xj_norm = np.sqrt(np.sum(xj**2))
            if xj_norm < 1e-10:
                continue

            best_r = -2.0
            best_lag = 0
            for lag in range(-max_lag, max_lag + 1):
                # Positive lag: shift j forward (i leads j)
                if lag >= 0:
                    seg_i = xi[:T - lag] if lag > 0 else xi
                    seg_j = xj[lag:] if lag > 0 else xj
                else:
                    seg_i = xi[-lag:]
                    seg_j = xj[:T + lag]

                if len(seg_i) < 3:
                    continue

                denom = np.sqrt(np.sum(seg_i**2) * np.sum(seg_j**2))
                if denom > 1e-10:
                    r = float(np.sum(seg_i * seg_j) / denom)
                else:
                    r = 0.0

                if r > best_r:
                    best_r = r
                    best_lag = lag

            opt_lag[i, j] = best_lag
            peak_corr[i, j] = round(best_r, 4)

    # Extract directed pairs (i leads j when lag > 0)
    directed = []
    for i in range(n):
        for j in range(n):
            if i != j and opt_lag[i, j] > 0 and peak_corr[i, j] > 0.3:
                directed.append((
                    labels[i], labels[j],
                    int(opt_lag[i, j]),
                    round(float(peak_corr[i, j]), 4),
                ))

    return {
        'optimal_lag': opt_lag,
        'peak_correlation': peak_corr,
        'labels': labels,
        'directed_pairs': directed,
    }


def build_adjacency(correlation_matrix_result, threshold=0.5,
                     method='absolute'):
    """Convert correlation matrix to binary adjacency matrix.

    Args:
        correlation_matrix_result: dict from correlation_matrix().
        threshold: Correlation threshold for edge creation.
        method: 'absolute' (|r| > threshold), 'positive' (r > threshold),
            or 'percentile' (top N% of correlations, threshold=percentile).

    Returns:
        dict with:
            adjacency: 2D binary array (n, n).
            n_edges: int, number of edges.
            labels: list of neuron IDs.
            edge_list: list of (source, target, weight) tuples.
    """
    corr = correlation_matrix_result['matrix']
    labels = correlation_matrix_result['labels']
    n = corr.shape[0]

    if method == 'percentile':
        mask = np.triu_indices(n, k=1)
        vals = np.abs(corr[mask])
        if len(vals) > 0:
            cutoff = np.percentile(vals, threshold)
        else:
            cutoff = 1.0
        adj = (np.abs(corr) >= cutoff).astype(int)
    elif method == 'positive':
        adj = (corr > threshold).astype(int)
    else:  # absolute
        adj = (np.abs(corr) > threshold).astype(int)

    np.fill_diagonal(adj, 0)

    edge_list = []
    for i in range(n):
        for j in range(n):
            if adj[i, j]:
                edge_list.append((labels[i], labels[j], round(float(corr[i, j]), 4)))

    return {
        'adjacency': adj,
        'n_edges': int(adj.sum()),
        'labels': labels,
        'edge_list': edge_list,
    }


def cascade_connectivity(traces, stim_neuron, baseline_frames=None,
                          threshold_dff=0.5, synaptic_delay=1):
    """Infer directed connectivity from a stimulation cascade.

    After stimulating one neuron, observe which others fire and when.
    Layer analysis determines the shortest path from the stimulated neuron
    to each responder. Direct connections are inferred from 1-step delays.

    Args:
        traces: 2D array (n_neurons, n_frames) where frame 0 is the
            stimulation frame.
        stim_neuron: int, index of the stimulated neuron.
        baseline_frames: indices for baseline; if None, uses stim_neuron's
            pre-stim value or frame 0 of non-stim neurons.
        threshold_dff: ΔF/F threshold to consider a neuron "fired".
        synaptic_delay: int, expected frames between presynaptic firing
            and postsynaptic response (default 1).

    Returns:
        dict with:
            first_fire: dict mapping neuron_id -> frame when it first fired.
            layers: dict mapping layer_number -> list of neuron IDs.
            cascade_tree: list of (source, target) edges inferred from cascade.
            re_stimulations: list of (neuron_id, frame) where neurons re-peak.
            n_responding: int, number of neurons that responded.
    """
    traces = np.asarray(traces, dtype=np.float64)
    n_neurons, n_frames = traces.shape

    # Compute baselines: use the minimum value for each neuron trace
    # (resting level, before any activity)
    baselines = np.zeros(n_neurons)
    if baseline_frames is not None:
        for i in range(n_neurons):
            baselines[i] = traces[i, baseline_frames].mean()
    else:
        # Use minimum value as baseline (resting state)
        for i in range(n_neurons):
            baselines[i] = float(np.min(traces[i]))

    # Detect first firing frame for each neuron
    first_fire = {}
    for i in range(n_neurons):
        bl = baselines[i]
        for t in range(n_frames):
            val = traces[i, t]
            if bl > 1e-6:
                dff = (val - bl) / bl
            else:
                dff = val - bl
            if dff > threshold_dff:
                first_fire[i] = t
                break

    # Build layers from first_fire times
    layers = {}
    for neuron, frame in first_fire.items():
        layer = frame // synaptic_delay
        if layer not in layers:
            layers[layer] = []
        layers[layer].append(neuron)

    # Infer cascade tree edges
    # Neurons at layer L+1 are targets of neurons at layer L
    cascade_tree = []
    sorted_layers = sorted(layers.keys())
    for idx in range(len(sorted_layers) - 1):
        src_layer = sorted_layers[idx]
        dst_layer = sorted_layers[idx + 1]
        sources = layers[src_layer]
        targets = layers[dst_layer]
        for dst in targets:
            # Assign to a source - if only one source in prev layer, clear
            if len(sources) == 1:
                cascade_tree.append((sources[0], dst))
            else:
                # Multiple possible sources - record all candidates
                cascade_tree.append((sources, dst))

    # Detect re-stimulations (neurons that re-peak after initial decay)
    decay_rate = 0.80  # typical calcium decay per frame
    re_stims = []
    for i in range(n_neurons):
        ff = first_fire.get(i)
        if ff is None:
            continue
        peak_val = traces[i, ff]
        bl = baselines[i]
        for t in range(ff + 2, n_frames):
            prev = traces[i, t - 1]
            actual = traces[i, t]
            # Re-peak: significant increase after decay started
            expected_decay = bl + (prev - bl) * decay_rate
            if actual > expected_decay * 1.2 and actual > bl + (peak_val - bl) * 0.3:
                # Confirm it's actually increasing (not just slow decay)
                if actual > prev * 1.1:
                    re_stims.append((i, t))
                    break  # Only first re-stimulation

    return {
        'first_fire': first_fire,
        'layers': {k: sorted(v) for k, v in layers.items()},
        'cascade_tree': cascade_tree,
        're_stimulations': re_stims,
        'n_responding': len(first_fire),
    }


def graph_metrics(adjacency, directed=True):
    """Compute graph-theoretic metrics from an adjacency matrix or dict.

    Args:
        adjacency: 2D binary array (n, n) or dict mapping node -> list of targets.
        directed: bool. If True, compute in/out degree separately.

    Returns:
        dict with:
            n_nodes: int.
            n_edges: int.
            in_degree: dict mapping node -> in-degree.
            out_degree: dict mapping node -> out-degree.
            total_degree: dict mapping node -> in + out degree.
            hub: node ID with highest total degree.
            hub_degree: int, total degree of hub.
            density: float, n_edges / (n_nodes * (n_nodes - 1)).
            reciprocity: float, fraction of edges with reverse edge (directed).
            clustering: dict mapping node -> local clustering coefficient.
            mean_clustering: float.
    """
    # Convert dict adjacency to matrix
    if isinstance(adjacency, dict):
        nodes = set(adjacency.keys())
        for tgts in adjacency.values():
            nodes.update(tgts)
        nodes = sorted(nodes)
        n = len(nodes)
        node_idx = {node: i for i, node in enumerate(nodes)}
        mat = np.zeros((n, n), dtype=int)
        for src, tgts in adjacency.items():
            for dst in tgts:
                mat[node_idx[src], node_idx[dst]] = 1
    else:
        mat = np.asarray(adjacency, dtype=int)
        n = mat.shape[0]
        nodes = list(range(n))
        node_idx = {i: i for i in range(n)}

    n_edges = int(mat.sum())

    # Degree computation
    in_deg = {}
    out_deg = {}
    total_deg = {}
    for i, node in enumerate(nodes):
        out_d = int(mat[i].sum())
        in_d = int(mat[:, i].sum())
        out_deg[node] = out_d
        in_deg[node] = in_d
        total_deg[node] = in_d + out_d

    # Hub
    hub = max(nodes, key=lambda n: total_deg[n])
    hub_degree = total_deg[hub]

    # Density
    max_edges = n * (n - 1) if directed else n * (n - 1) // 2
    density = n_edges / max_edges if max_edges > 0 else 0.0

    # Reciprocity (fraction of edges that are bidirectional)
    if directed and n_edges > 0:
        reciprocal = 0
        for i in range(n):
            for j in range(n):
                if mat[i, j] and mat[j, i]:
                    reciprocal += 1
        reciprocity = reciprocal / n_edges
    else:
        reciprocity = 0.0

    # Local clustering coefficient
    clustering = {}
    for i, node in enumerate(nodes):
        if directed:
            neighbors = set()
            for j in range(n):
                if mat[i, j] or mat[j, i]:
                    neighbors.add(j)
        else:
            neighbors = {j for j in range(n) if mat[i, j]}

        k = len(neighbors)
        if k < 2:
            clustering[node] = 0.0
            continue

        # Count edges between neighbors
        neighbor_edges = 0
        neighbor_list = list(neighbors)
        for a in range(len(neighbor_list)):
            for b in range(len(neighbor_list)):
                if a != b and mat[neighbor_list[a], neighbor_list[b]]:
                    neighbor_edges += 1

        max_neighbor_edges = k * (k - 1)
        clustering[node] = round(neighbor_edges / max_neighbor_edges, 4) if max_neighbor_edges > 0 else 0.0

    mean_clustering = float(np.mean(list(clustering.values()))) if clustering else 0.0

    return {
        'n_nodes': n,
        'n_edges': n_edges,
        'in_degree': in_deg,
        'out_degree': out_deg,
        'total_degree': total_deg,
        'hub': hub,
        'hub_degree': hub_degree,
        'density': round(density, 4),
        'reciprocity': round(reciprocity, 4),
        'clustering': clustering,
        'mean_clustering': round(mean_clustering, 4),
    }
