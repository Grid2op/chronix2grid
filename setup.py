from setuptools import setup

extras = {
    "docs": ["numpydoc", "sphinx", "sphinx_rtd_theme", "sphinxcontrib_trio"],
    "test": ["nbformat", "jupyter_client", "jyquickhelper"]
}

all_targets = []
for el in extras:
    all_targets += extras[el]
extras["all"] = list(set(all_targets))

setup(name='Chronix2Grid',
      version='0.1.0',
      description='A python package to generate "en-masse" chronics for loads and productions (thermal, renewable)',
      long_description='TODO',
      classifiers=[
          'Development Status :: 5 - ALPHA',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          "License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)",
          "Intended Audience :: Developers",
          "Intended Audience :: Education",
          "Intended Audience :: Science/Research",
          "Natural Language :: English"
      ],
      keywords='ML powergrid optmization RL power-systems chronics generation production load network',
      author='Mario Jothy, Nicolas Megel, Vincent Renault',
      author_email=' mario.jothy@artelys.com',
      url="https://github.com/mjothy/chronix2grid",
      license='Mozilla Public License 2.0 (MPL 2.0)',
      packages=['chronix2grid'],
      include_package_data=True,
      install_requires=[],
      zip_safe=False,
      entry_points={'console_scripts': ['chronix2grid.main=grid2viz.command_line:main']}
)