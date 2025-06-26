# `name` is the name of the package as used for `pip install package`
name = "temporalviz"
# `path` is the name of the package for `import package`
path = name.lower().replace("-", "_").replace(" ", "_")
# Your version number should follow https://python.org/dev/peps/pep-0440 and
# https://semver.org
version = "0.1.dev0"
author = "Charles Xu"
author_email = "charlsxu@mit.edu"
description = "Package for visualization of temporal data"  # One-liner
url = "https://github.com/HungryAmoeba/cow_viz"  # your project homepage
license = "MIT"  # See https://choosealicense.com
