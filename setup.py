from setuptools import setup, find_packages
import versioneer


def readme():
    with open("README.md") as f:
        return f.read()


setup(name="iceburger",
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      description=("Library for Iceburger Classification Challenge"),
      long_description=readme(),
      author=["Hidy Chiu, Wei-Yi Cheng"],
      author_email=["hidy0503@gmail.com", "ninpy.weiyi@gmail.com"],
      classifiers=[
          "Development Status :: 3 - Alpha",
          "Intended Audience :: Developers",
          "Programming Language :: Python :: 3.6"
      ],
      packages=find_packages(),
      install_requires=[
          "pandas",
          "keras>=2.0.9",
          "scikit-learn>=0.18.1",
          "h5py>=2.7.0"
      ],
#     extras_require={"test": ["nose", "nose-parameterized>=0.5", "mock"]},
#     package_data={"deepsense": ["resources/templates/*",
#                                 "resources/har_model/*"]},
      entry_points={
          "console_scripts": [
              "iceburger-conv2d-train=iceburger.conv_train:main",
	      "iceburger-inception-train-small=iceburger.inception_train:main",
              "iceburger-inception-train=iceburger.inception_imagenet_train:main",
              "iceburger-resnet20-train=iceburger.resnet20_train:main",
              "iceburger-predict=iceburger.predict:main",
              "iceburger-fconv2d-train=iceburger.fconv_train:main",
              "iceburger-inception-train-aws=iceburger.inception_imagenet_train_aws:main",
              "iceburger-composite-train=iceburger.composite_model:main"
           ]
      },
      include_package_data=True,
      zip_safe=False)
