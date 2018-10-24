#!/usr/bin/env python

from setuptools import setup

if __name__ == '__main__':
    setup(
        name='BirdRoostDetection',
        version='1.0',
        description='Machine Learning to locate pre migratory  bird roosts in '
                    'NEXRAD radar data',
        author='Carmen Chilson',
        author_email='carmenchilson@ou.edu',
        license='MIT',
        url='https://github.com/carmenchilson/BirdRoostDetection',
        packages=[
            'BirdRoostDetection',
            'BirdRoostDetection.BuildModels',
            'BirdRoostDetection.BuildModels.ShallowCNN',
            'BirdRoostDetection.BuildModels.Inception',
            'BirdRoostDetection.BuildModels.Part_Of_Image',
            'BirdRoostDetection.PrepareData',
            'BirdRoostDetection.ReadData',
            'BirdRoostDetection.Analysis'
        ],
        package_data={'BirdRoostDetection': ['settings.json']},
        keywords=[
            'Machine Learning',
            'Biology',
            'Tensorflow',
            'keras',
            'GANS',
            'ConvNet',
            'CNN',
            'Convolutional Neural Network',
            'Bird Roost',
            'Aeroecology'
        ],
        classifiers=[
            'Intended Audience :: Biology Research using Computer Science',
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python :: 2.7'
        ]
    )
