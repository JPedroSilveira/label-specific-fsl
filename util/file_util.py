import os

def create_folder_if_not_exists(folder_path, root_path=None):
  final_path = folder_path
  if root_path:
    final_path = os.path.join(root_path, final_path)
  if not os.path.exists(final_path):
    os.makedirs(final_path)