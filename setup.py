from setuptools import setup, find_packages

setup(name='tensorflow-training',
      version='0.0.1',
      description='A few practical examples for tf training.',
      long_description='',
      classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Operating System :: Unix',
        'Programming Language :: Python',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6'
      ],
      keywords='tensorflow training',
      url='https://github.com/blazekadam/tensorflow-training',
      author='Adam Blazek',
      author_email='blazekada@gmail.com',
      license='MIT',
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False,
      install_requires=['tensorflow>=1.7', 'numpy', 'pytest', 'tensorboard'])
