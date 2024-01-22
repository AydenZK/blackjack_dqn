from setuptools import setup, find_namespace_packages

setup(
    name="blackjack_dqn",
    version="0.1.0",
    author="Ayden Khalil",
    author_email="aydenkhalil619@gmail.com",
    description="DQN for Blackjack",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/AydenZK/blackjack_dqn",
    license="MIT",
    package_dir={"": "src"},
    packages=find_namespace_packages(where="src"),
)