import os
import shutil

import click
import logging
import zipfile
import requests
import torchaudio
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

logger = logging.getLogger(__name__)


def list_files_from_zenodo(record_id):
    base_url = "https://zenodo.org/api/records/"
    response = requests.get(f"{base_url}{record_id}")
    if response.status_code == 200:
        data = response.json()
        files = data['files']
        return [(file['key'], file['links']['self'], int(file['size'])) for file in files]
    else:
        raise RuntimeError("Unable to access Zenodo record")


def download_file(url, path_to_save):
    response = requests.get(url)
    if response.status_code == 200:
        path_to_save.parent.mkdir(parents=True, exist_ok=True)
        with open(path_to_save, 'wb') as f:
            f.write(response.content)
    else:
        raise RuntimeError("Unable to download the file")


@click.command()
@click.argument('output_filepath', type=click.Path())
def main(output_filepath: Path) -> None:
    """ Download the required datasets from the web (SPEECHCOMMANDS and DEMAND)
        if not already downloaded and save them to the specified output_filepath.
    """
    output_filepath = Path(output_filepath)

    logger.info('downloading SPEECHCOMMANDS')
    speechcmd_subfolder = Path('SPEECHCMD')
    path_to_save = output_filepath / speechcmd_subfolder
    path_to_save.mkdir(parents=True, exist_ok=True)
    ds = torchaudio.datasets.SPEECHCOMMANDS(root=path_to_save, subset=None, download=True)
    logger.info(f"Dataset contains {len(ds)} datapoints")

    logger.info('downloading DEMAND')
    record_id = '1227121'
    files_list = list_files_from_zenodo(record_id)
    files_list = sorted(files_list, key=lambda x: x[2])
    files_list = [f for f in files_list if '16k' in f[0]]

    zip_subfolder = Path('DEMAND/zips')
    logger.info(f"Downloading {len(files_list)} files")
    for file_name, url, _ in files_list:
        path_to_save = output_filepath / zip_subfolder / file_name
        if not path_to_save.exists():
            logger.info(f"Downloading {file_name} from {url} to {path_to_save}")
            download_file(url, path_to_save)

    wavs_subfolder = Path('DEMAND/wavs')
    paths_to_unzip = [path for path in (output_filepath / zip_subfolder).iterdir() if path.suffix == '.zip']
    logger.info(f"Unzipping {len(paths_to_unzip)} files")
    for zip_path in paths_to_unzip:
        path_to_extract = output_filepath / wavs_subfolder
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            if not path_to_extract.exists():
                logger.info(f"Unzipping {zip_path} to {path_to_extract}")
                path_to_extract.mkdir(parents=True, exist_ok=True)
                zip_ref.extractall(path_to_extract)

    logger.info('Deleting zip files')
    # delete demand zip files
    for zip_path in paths_to_unzip:
        zip_path.unlink()
    (output_filepath / zip_subfolder).absolute().rmdir()
    # delete speechcmd zip file
    speechcmd_archive = [a for a in (output_filepath / speechcmd_subfolder).iterdir() if a.suffix == '.gz']
    if len(speechcmd_archive) == 1:
        speechcmd_archive[0].unlink()

    # Zip the DEMAND and SPEECHCMD directories into a single file (this file will be versioned using DVC)
    logger.info('Zipping DEMAND and SPEECHCMD directories')
    with zipfile.ZipFile(output_filepath / 'data.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
        for directory in [output_filepath / 'DEMAND', output_filepath / 'SPEECHCMD']:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    zipf.write(
                        os.path.join(root, file),
                        os.path.relpath(os.path.join(root, file), os.path.join(directory, '..'))
                    )

    # Remove the original directories
    logger.info('Removing original DEMAND and SPEECHCMD directories')
    shutil.rmtree(output_filepath / 'DEMAND')
    shutil.rmtree(output_filepath / 'SPEECHCMD')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
