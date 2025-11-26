TemporalViz
===========

Scientific temporal visualization with Matplotlib and optional Blender backends.

- **Matplotlib backend**: fast, headless-friendly animations (2D/3D, quivers, graphs)
- **Blender backend (optional)**: high‑fidelity 3D renders via Blender

Install
-------

.. code-block:: bash

    pip install temporalviz

With Blender extras:

.. code-block:: bash

    pip install "temporalviz[blender]"

From source:

.. code-block:: bash

    python -m venv .venv && source .venv/bin/activate
    pip install -U pip
    pip install -e .

Quick start
-----------

.. code-block:: python

    import numpy as np
    from temporalviz import visualize_dynamics

    # Brownian motion in 2D
    T, N = 200, 50
    steps = 0.15 * np.random.randn(T, N, 2)
    pos = steps.cumsum(axis=0)

    # Save an animation (Matplotlib) - requires ffmpeg installed
    anim = visualize_dynamics("matplotlib", pos, save_path="out.mp4")

CLI
---

Run built-in examples:

.. code-block:: bash

    temporalviz-examples brownian --save out.mp4
    temporalviz-examples graph

API
---

- ``temporalviz.create_visualizer(config)`` → returns a visualizer instance
  - ``config`` may be ``"matplotlib"`` or ``"blender"`` or a dict such as
    ``{"backend": "matplotlib", "interval": 40}``
- ``temporalviz.visualize_dynamics(config, pos, ori=None, save_path=None, **kwargs)``
  - ``pos`` shape ``(T, N, D)`` with ``D ∈ {2, 3}``
  - ``ori`` optional: ``(T, N, D)`` direction vectors or ``(T, N, 4)`` quaternions (w,x,y,z)

Graphs (Matplotlib)
-------------------

Overlay edges using a ``networkx`` graph:

.. code-block:: python

    import numpy as np, networkx as nx
    from temporalviz import create_visualizer

    T, N = 100, 16
    x = np.linspace(0, 1, N)
    t = np.linspace(0, 4*np.pi, T)
    pos = np.zeros((T, N, 2))
    pos[:, :, 0] = x[None, :]
    pos[:, :, 1] = 0.2*np.sin(t[:, None] + 2*np.pi*x[None, :])

    G = nx.path_graph(N)
    viz = create_visualizer({"backend": "matplotlib", "interval": 40})
    anim = viz.visualize(pos, graph=G)

Blender backend (optional)
--------------------------

Exports trajectories to a temp file and invokes Blender headless for rendering.
Ensure Blender is installed; set the executable if needed (macOS example shown):

.. code-block:: python

    from temporalviz import visualize_dynamics
    cfg = {"backend": "blender", "fps": 24,
           "blender_exec": "/Applications/Blender.app/Contents/MacOS/Blender"}
    anim = visualize_dynamics(cfg, pos, save_path="render.mp4")

Advanced Blender options
------------------------

- ``primitive_shape``: ``"sphere"`` (default), ``"cone"``, or ``"cube"``
- ``primitive_color``: ``[R, G, B]`` in [0,1] (static); or pass ``node_colors`` arrays
- ``node_colors``: ``(T,N,3/4)`` or ``(N,3/4)`` for per-frame/per-node colors
- ``node_sizes``: ``(T,N)`` or ``(N,)`` for per-frame/per-node sizes
- ``ori``: ``(T,N,3)`` direction vectors (auto-aligned) or ``(T,N,4)`` quaternions (w,x,y,z)
- ``graph``: ``networkx`` graph or edge array ``(E,2)``; edges are animated in 3D
- ``edge_bevel``: curve thickness for edges (e.g., ``0.015``)

Examples
--------

- ``examples/brownian_motion.py``: 2D Brownian motion (MP4 in ``examples/output/``)
- ``examples/path_graph_wiggle.py``: path graph wiggle (MP4 in ``examples/output/``)
- Blender demos in ``examples/`` (require Blender)

Testing
-------

.. code-block:: bash

    # Run the test suite (uses a headless Matplotlib backend)
    MPLBACKEND=Agg pytest -q

Changelog
---------

See ``CHANGELOG.rst``.

