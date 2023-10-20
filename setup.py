from setuptools import setup,find_packages
  
NAME="MLPROJECT_for_part _tracking"
VERSION="0.0.1"
DESC="ML project to predict parts which will fail in automotive"
AUTHOR="hrishikesh"
AUTHOR_EMAIL="hrishikeshbhagawati@gmail.com"

REQUIREMENTS="requirements.txt"
HYPHEN_E_DOT="-e ."

def get_requirements():
    with open(REQUIREMENTS) as req_file:
        req_file=req_file.readlines()

        req_list=[i.replace("\n","") for i in req_file]
    
        if HYPHEN_E_DOT in req_list:
            req_list.remove(HYPHEN_E_DOT)
    return req_list


setup( 
    name=NAME, 
    version=VERSION, 
    description=DESC, 
    author=AUTHOR, 
    author_email=AUTHOR_EMAIL, 
    packages=find_packages(), 
    install_requires=get_requirements() 
) 