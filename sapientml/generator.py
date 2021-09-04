import os
import requests
import polling
import shutil
import secrets
import hashlib
import pandas as pd

server = '3.224.119.219'
sig_table = {
    "05995cb9f3f2d12449ddc0b8c0a337c6a1cd6b65", # "Kobe-Bryant-Shot-Selection"
    "a56874e0b81b8aa2ad6bbc6ab79c44a461bf8323", # "sentiment-analysis-on-movie-reviews"
    "2c90873a2fc09c7914a14d521fa1e09a939370a3", # "sf1"
    "a368ac8818b5f00a8b186c9bcd5954c9d7dca1dd", # "917"
    "1d6e8605059a7011041ec91188051171cfb05fff", # "1049"
    "93d5647d699d1eead3a346f953f7f38b5063e4e0", # "1120"
    "0ad2ea5a30cb85e68c50898946faf2287399a947", # "costa-rica"
    "4089224097ce81d20c71580a69cce2f74bf1496b", # "wine_quality_white"
    "812524ad61fdef97ca33d1bb2f249a3bffc7fdb6", # "mercedes-benz"
    "203087fdf1e05a18b0be6e8ca9da9b01166cfd80", # "spooky-author-identification"
    "e391cb0647e858d4011cf295ab2d963df341f1b8", # "38_new"
    "bf24836ee82fd8e71a3f88446a51843523e48b8e", # "housing-prices"
    "69ba8c5484fc0029c6c6be6cdfed64bd8e1938fc", # "jura"
    "8c1b99b315842f185088183736687a535ac919ed", # "Hill_Valley_with_noise"
    "1f0a496dc1bcfc8ff2c88936490cb73290cc6c22", # "ionosphere"
    "e391cb0647e858d4011cf295ab2d963df341f1b8", # "38"
    "f815b6b2476ea0d2eda7535ea6ed0ae0c7f70e10", # "glass"
    "903aaaa0b782b888e0ec5a9b0b6bd0c30b179968", # "car_evaluation"
    "eba74de1cf0722d5b4fccff25a3c0d9c44ca8bb0", # "spambase"
    "11537e44ace47ada074f95cb57c1ae957afc257c", # "Categorical-Feature-Encoding-Challenge-II"
    "1092160ac088912ecc3e4878cd94de8114c18351", # "389"
    "4efc23d72219c83528d5ee273b478a8c08d97e6f", # "titanic"
    "cc977eea9f4569cd100afa366afb5c644d5f5b4a", # "184"
    "6836e3feca6f692a55428e9b0822b7412418a626", # "enb"
    "7374bbe2f8dc562c8c90dfb908385afe7287c081", # "detecting-insults-in-social-commentary"
    "8cbba4d5abddac7c01987604b262e4af151514d2", # "breast_cancer_wisconsin"
    "d95ce073d4b2e316ed5c834fe4e8168897f7e10e", # "wine_quality_red"
    "96bda8eeefe2a848b57eba65878476ce384110a6", # "179"
    "38d7ffea4d1df0fcc0524bbef14e401f69570f69", # "sf2"
    "a906cd92507afd5eb09a856b805dc1e91d47083e", # "Hill_Valley_without_noise"
    "42193998858fbe41d15576dd7046bf30c90bae34", # "46"
    "7893be09f8343b43c99b69768747b86e721a9dd6", # "772"
}

class SapientMLGenerator:
    def __init__(self):
        pass

    def generate_code(self, dataframe, random_seed_id=1):
        print("   Info: Generating pipeline code. This may take several minutes...")

        if random_seed_id < 1 or random_seed_id > 10:
            print("  Error: Random seed ID must be between 1 and 10.")
            return

        if not self.check_df(dataframe):
            print("  Error: This version of SapientML can take only the datasets under benchmarks directory.")
            return

        try:
            session_id = self.generate_session_id()
            self.run_sapientml(session_id, dataframe, random_seed_id)
            print("   Info: Pipeline code is generated under './outputs' directory.")
        except Exception as e:
            print("  Error: Exception occured during running SapientML:")
            print(e)
            if os.path.exists('./outputs'):
                shutil.rmtree('./outputs')
            return
        finally:
            if os.path.exists('./tmp'):
                shutil.rmtree('./tmp')

    def check_df(self, dataframe):
        sig = hashlib.sha1(pd.util.hash_pandas_object(dataframe).values).hexdigest()
        return sig in sig_table

    def run_sapientml(self, session_id, dataframe, random_seed_id):
        if os.path.exists('./tmp'):
            shutil.rmtree('./tmp')
        os.makedirs('./tmp', exist_ok=True)

        dataframe.to_pickle('./tmp/input.pkl.bz2')

        url_new = f"http://{server}/{session_id}/{str(random_seed_id)}/new"
        input_file = open("./tmp/input.pkl.bz2", "rb")
        resp = requests.post(url_new, files = {"file": input_file})

        url_done = f"http://{server}/{session_id}/done"
        resp = polling.poll(lambda: requests.get(url_done).status_code == 200, step=10, poll_forever=True)

        url_outputs = f"http://{server}/{session_id}/outputs.tar.bz2"
        resp = requests.get(url_outputs)

        if resp.status_code != 200:
            raise RuntimeError('No outputs generated.')

        tarfile_name = './tmp/outputs.tar.bz2'
        with open(tarfile_name, 'wb') as f:
            f.write(resp.content)

        if os.path.exists('./outputs'):
            shutil.rmtree('./outputs')
        shutil.unpack_archive(tarfile_name)

        logfile = './outputs/sapientml.log'
        if os.path.exists(logfile):
            with open(logfile, "r") as f:
                print(f.read(), end="")

    def generate_session_id(self):
        return secrets.token_urlsafe(16)
