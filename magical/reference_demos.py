"""Tools for downloading reference demonstrations."""
import argparse
import datetime
import logging
import os
import tempfile
import zipfile

import requests

__all__ = ['try_download_demos']

REFERENCE_DEMO_ZIP = "https://github.com/qxcv/magical-data/archive/master.zip"
REFERENCE_DEMO_ZIP_ROOT = "magical-data-master"
DEFAULT_LOCATION = "demos"
DONE_FILE = ".download-done"


def try_download_demos(dest=DEFAULT_LOCATION, progress=True):
    """Download demonstrations to 'dest', if they do not already exist.

    Args:
        dest (str): destination directory to write demonstrations to.
        progress (bool): should progress messages be displayed to stdout?"""
    zip_url = REFERENCE_DEMO_ZIP
    done_path = os.path.join(dest, DONE_FILE)

    if os.path.exists(done_path):
        # already downloaded
        logging.info(f"Demonstrations appear to already be in '{dest}'; to "
                     "force download, delete that directory and try again")
        return

    with tempfile.NamedTemporaryFile("wb") as temp_fp:
        # download zipped copy of master
        logging.info(
            f"Destination not found. Downloading data from f'{zip_url}' to a "
            "temporary file")
        mb_downloaded = _download_file(zip_url, temp_fp, progress=progress)
        temp_fp.flush()
        logging.info(f"Download done, got {mb_downloaded:.2f}MB")

        # extract each file in the zipped copy to destination directory
        _recursive_extract(zip_fp=temp_fp.name,
                           dest_dir=dest,
                           member_prefix=REFERENCE_DEMO_ZIP_ROOT)

    # create a file to indicate that we have already downloaded this data
    with open(done_path, "w") as fp:
        print(
            f"Data downloaded from '{zip_url}' on "
            f"{datetime.datetime.now().isoformat()}",
            file=fp)


class DownloadError(Exception):
    pass


def _download_file(url,
                   dest_fp,
                   progress_mb=10,
                   progress=True,
                   bufsize=32 * 1024):
    try:
        # launch request & make sure we get a non-error code
        res = requests.get(url, stream=True)
        res.raise_for_status()

        # TODO: infer content size. It's not as simple as looking at
        # Content-Length, since response seems to have Transfer-Encoding:
        # chunked. Somehow browsers do this anyway, though?

        # download one chunk at a time, logging as we go
        next_report_mb = 0
        bytes_downloaded = 0
        for chunk in res.iter_content(bufsize):
            # read one chunk
            dest_fp.write(chunk)
            bytes_downloaded += len(chunk)

            # logging
            mb_downloaded = bytes_downloaded / float(1024**2)
            if progress and mb_downloaded >= next_report_mb:
                print(f'{mb_downloaded:.2f}MiB downloaded')
                next_report_mb += progress_mb

        return mb_downloaded

    except (IOError, requests.HTTPError) as ex:
        raise DownloadError(f"Error while downloading '{url}': {ex}")


def _recursive_extract(zip_fp, dest_dir, member_prefix=None, overwrite=True):
    """Recursively extract a directory from a zip file to another directory on
    the filesystem, overwriting files as necessary."""

    # we're going to use these to figure out relative paths
    if member_prefix is not None:
        split_zip_path = member_prefix.split(os.path.sep)
        prefix_len = len(split_zip_path)
    else:
        split_zip_path = ()
        prefix_len = 0

    with zipfile.ZipFile(zip_fp, 'r') as zip_fp:
        # now iterate over each member of the zip file, and extract any members
        # that are (1) actually files (not directories), and (2) fall under
        # member_prefix in the zip file
        for zip_rel_path in zip_fp.namelist():
            split_rel_path = zip_rel_path.split(os.path.sep)
            if split_rel_path and split_rel_path[-1] == '':
                # skip directories (that end with a "/")
                continue
            if split_rel_path[:len(split_zip_path)] != split_zip_path:
                # skip directories that don't begin with 'member_prefix'
                continue

            # make the directory that we will write this file to
            rel_parts = os.path.sep.join(split_rel_path[prefix_len:])
            out_path = os.path.join(dest_dir, rel_parts)
            save_dir = os.path.dirname(out_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)

            # write the file
            logging.debug(f"Extracting '{zip_rel_path}' -> '{out_path}'")
            with open(out_path, 'wb') as fs_fp:
                fs_fp.write(zip_fp.read(zip_rel_path))


def _main():
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser(
        description="Download MAGICAL reference demonstrations")
    parser.add_argument(
        "--dest",
        help=f"destination to download to (default: '{DEFAULT_LOCATION}')",
        default=DEFAULT_LOCATION,
        type=str)
    args = parser.parse_args()

    try_download_demos(dest=args.dest)


if __name__ == '__main__':
    _main()
