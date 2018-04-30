import os
import shutil
import urllib.request


def download_data() -> None:
    for name, filename in (('facial-keypoints-train.zip', 'training.zip'), ('facial-keypoints-test.zip', 'test.zip')):
        print('Downloading {} to {}'.format(name, filename))
        urllib.request.urlretrieve('https://blazeka.cz/'+name, filename)
        print('Extracting {}'.format(filename))
        shutil.unpack_archive(filename, '.')
        os.remove(filename)
    print('done')


if __name__ == '__main__':
    download_data()
