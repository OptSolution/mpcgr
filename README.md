# Intro
Practical Semi-automatic Global Registration of Multiple Point Clouds Based on Semidefinite Programming

# Dependence
- Open3d 0.12.0
- igraph
- cvxpy

# Command
If you only want to test, you can only run:
```
python run.py
```

If you want to run on your own data, you can run:
```
python run.py [data_path] [config_file_path]
```

We provide some config file in the `config` folder. In the `[data_path]`, you should put a folder named `data` in it, and put all the point cloud files in the `data` folder. You can refer to the structure of the `test_data` folder.

# Note (Manual operation)
Pairwise registration will be performed first, you need to wait for a while.

**When the pairwise registration visualization window pops up**:
* If the registration failed or the result, *you need press `D` to delete the correspondences and press `Esc` to close the window*
* If the registration succeeded, *you can press `Esc` directly*  