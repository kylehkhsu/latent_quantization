import pathlib
import shutil
import h5py


def get_data_path(config):
    file_name = pathlib.Path(config.storage.data_file_name)
    nfs_dir = pathlib.Path(config.storage.nfs_dir)
    nfs_path = nfs_dir / file_name

    use_local = config.storage.use_local
    possible_local_dirs = config.storage.possible_local_dirs

    if use_local:
        print('using local storage')
        while len(possible_local_dirs) > 0:
            local_dir = pathlib.Path(possible_local_dirs.pop(0))
            try:
                local_dir.mkdir(parents=True, exist_ok=True)
                print(f'located/created local cache directory {local_dir}')
                break
            except:
                print(f'could not create local cache directory {local_dir}')
                pass
        local_path = local_dir / file_name

        if local_path.exists():
            print('local storage found')
            nfs_timestamp = h5py.File(nfs_path, 'r').attrs['timestamp']
            print(f'nfs_timestamp: {nfs_timestamp}')
            local_timestamp = h5py.File(local_path, 'r').attrs.get('timestamp')
            print(f'local timestamp: {local_timestamp}')
            if nfs_timestamp != local_timestamp:
                print('local cache outdated; copying data to local storage')
                shutil.copy(nfs_path, local_path)
            else:
                print('local cache found and up-to-date')
        else:
            print('local cache not found; copying data to local storage')
            local_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(nfs_path, local_path)
        return local_path
    else:
        return nfs_path
