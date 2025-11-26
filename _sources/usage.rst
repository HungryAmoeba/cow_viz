Usage
=====

Installation
------------

.. code-block:: bash

    pip install temporalviz

Blender support (optional):

.. code-block:: bash

    pip install "temporalviz[blender]"

Or from source:

.. code-block:: bash

    git clone https://github.com/HungryAmoeba/cow_viz
    cd cow_viz
    pip install -e .

Quick start
-----------

.. code-block:: python

    import numpy as np
    from temporalviz import visualize_dynamics

    T, N = 200, 50
    steps = 0.15 * np.random.randn(T, N, 2)
    pos = steps.cumsum(axis=0)
    anim = visualize_dynamics("matplotlib", pos)

CLI quickstart
--------------

Run built-in examples:

.. code-block:: bash

    temporalviz-examples brownian --save out.mp4
    temporalviz-examples graph

Matplotlib configuration
------------------------

``create_visualizer`` accepts a dict with options, e.g. ``{"backend": "matplotlib", "interval": 40}``.

Supported keywords:

- ``interval``: milliseconds between frames
- ``title``: figure title
- ``draw_edges``: whether to draw graph edges
- ``auto_size``: rescale ``node_sizes`` to a nice range (default True)
- ``node_size_range``: min/max sizes when ``auto_size`` is True
- ``edge_opacity_by_distance``: fade edges with length
- ``edge_opacity_scale``: distance scale for fading
- ``edge_width``: line width
- ``xlabel``, ``ylabel``, ``zlabel``: axis labels

Blender backend
---------------

To render with Blender (optional), set ``{"backend": "blender"}`` and ensure Blender is installed.
The backend will export the trajectory and call Blender in background mode to render.

.. code-block:: python

    from temporalviz import visualize_dynamics
    anim = visualize_dynamics({"backend": "blender", "fps": 24, "save_path": "render.mp4"}, pos)

If Blender isn't auto-discovered, set the path explicitly:

.. code-block:: python

    cfg = {"backend": "blender", "blender_exec": "/Applications/Blender.app/Contents/MacOS/Blender"}
    visualize_dynamics(cfg, pos, save_path="render.mp4")

Advanced options (Blender)
--------------------------

- ``primitive_shape``: ``"sphere"`` | ``"cone"`` | ``"cube"``
- ``primitive_color``: static ``[R,G,B]``; or use dynamic ``node_colors`` arrays
- ``node_colors``: ``(T,N,3/4)`` or ``(N,3/4)`` (per-frame/per-node)
- ``node_sizes``: ``(T,N)`` or ``(N,)`` (per-frame/per-node)
- ``ori``: direction ``(T,N,3)`` (auto-aligned) or quaternion ``(T,N,4)`` (w,x,y,z)
- ``graph``: networkx graph or edge index array ``(E,2)`` (animated edges)
- ``edge_bevel``: edge thickness (e.g. ``0.015``)



