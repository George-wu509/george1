
Method 1: [add the following to your python code]

import ssl

ssl._create_default_https_context = ssl._create_unverified_context

Method 2: [under linux cmd]

$ sudo apt-get update

$ sudo apt-get install --reinstall ca-certificates

$ sudo update-ca-certificates

Method 3: [add trusted-host to pip command]

For example

$ pip install --trusted-host pypi.nvidia.com --extra-index-url [https://pypi.nvidia.com](https://pypi.nvidia.com) cuml-cu11

$ pip install --trusted-host pypi.org

Mehod 4: [upgrade certifi]

$ pip install --upgrade certifi

$ conda install -c anaconda certifi

Method 5: [add env variable]

set PYTHONHTTPSVERIFY=0

Method 6: [manually download certification from website]

down cacert.pem from curl website and put in folder such as  C:\path\to\cacert.pem

import ssl

import urllib.request

context = ssl.create_default_context(cafile="C:/path/to/cacert.pem")

response = urllib.request.urlopen("https://example.com", context=context)

print(response.read())

Method 5:

pip config set global.trusted-host "pypi.org files.pythonhosted.org pypi.python.org" --trusted-host=pypi.python.org --trusted-host=pypi.org --trusted-host=files.pythonhosted.org