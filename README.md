# Preparation - Mimic DB creation, Create derived tables, Generate Files to be imported by Python codes for analysis


1. Get MIMIC-iii files after goint thru approval process.

https://physionet.org/content/mimiciii/1.4/


2. Create Postgres DB that contain Mimic-iii data on Docker.  Go to the below site, and follow the instructions.

https://github.com/MIT-LCP/mimic-code/tree/main/mimic-iii/buildmimic/docker


3.  Preprocess the data in DB, and then generate the files that will be loaded by Python.  Go to the below site and follow instructions.

https://github.com/MLforHealth/MIMIC_Extract



# Here are the MAIN parts.


To set up Python environment, you may use virtual envs like Anaconda or may not.  Here are the versions that were used for the key libraries.  Please note that if you use different versions, you have get errors and may need to modify the codes.  Please note the Python version is different from the preparation phase.

gensim                    4.2.0            py39h1832856_0    conda-forge
nltk                      3.7                pyhd3eb1b0_0    anaconda
pandas                    1.4.3            py39h6a678d5_0    anaconda
numpy                     1.23.4                   pypi_0    pypi
python                    3.9.13          h9a8a25e_0_cpython    conda-forge
scikit-learn              1.1.2            py39he5e8d7e_0    conda-forge
scipy                     1.9.1            py39h8ba3f38_0    conda-forge
spacy                     3.1.6                    pypi_0    pypi
tensorflow                2.7.0           cpu_py39h4655687_0    conda-forge

Just run the below files sequentially.  Please note they will use the output files from the preparation 3 process.  Please note that the insstructions were referenced below site despite the difference in that the source codes have been modify and fixed properly in this version of Python and other libraries.  Please redirect the standard outputs to obtain the logs.

https://github.com/tanlab/ConvolutionMedicalNer

1. Run `python 01-Extract-Timeseries.py`

2. Copy the `ADMISSIONS.csv`, `NOTEEVENTS.csv`, `ICUSTAYS.csv` files into `data` folder.

3. Run `python 02-Select-Sub.py`.

4. Run `python 03-Preprocess.py`.

5. Install  `en_core_med7_lg` by `running pip install https://huggingface.co/kormilitzin/en_core_med7_lg/resolve/main/en_core_med7_lg-any-py3-none-any.whl`

6. Run `python 04-Apply.py`.

7. Get pretarined word2vec and fasttext models `word2vec.model` from below sites and `wiki-news-300d-1M.bin` and put them on `embedding` folder.

word2vec.model - https://github.com/kexinhuang12345/clinicalBERT
wiki-news-300d-1M.bin (fasttext) - https://fasttext.cc/docs/en/english-vectors.html

8. Run `python 05.py`.

9. Run `python 06.py`.

10. Run `python 06.py`.

11. Run `python 07-TimeSeries.py`.

12. Run `python 08.py`.

13. Run `python 09.py`.