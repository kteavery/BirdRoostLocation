#!/usr/bin/env python

from setuptools import setup

if __name__ == '__main__':
    setup(
        name='BirdRoostLocation',
        version='1.0',
        description='Machine Learning to locate pre migratory  bird roosts in '
                    'NEXRAD radar data',
        author='Kate Avery',
        author_email='katherine.avery@ou.edu',
        license='MIT',
        url='https://github.com/kteavery/BirdRoostLocation',
        packages=[
            'BirdRoostLocation',
            'BirdRoostLocation.BuildModels',
            'BirdRoostLocation.BuildModels.ShallowCNN',
            'BirdRoostLocation.BuildModels.Inception',
            'BirdRoostLocation.BuildModels.Part_Of_Image',
            'BirdRoostLocation.PrepareData',
            'BirdRoostLocation.ReadData',
            'BirdRoostLocation.Analysis'
        ],
        package_data={'BirdRoostLocation': ['settings.json']},
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
            'Programming Language :: Python :: 3.6'
        ]
    )
