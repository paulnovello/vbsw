import os
import sys
import datetime

def backup(dirname):
    path = os.path.dirname(os.path.realpath(__file__))
    os.system("mkdir " + path + "/../../../backups/backup" + dirname)
    os.system("cp -r " + path + "/../results " + path +
              "/../../../backups/backup" + dirname + "/")
    os.system("cp -r " + path + "/../postprocessing " +
              path + "/../../../backups/backup" + dirname + "/")
    os.system("cp -r " + path + "/../params " +
              path + "/../../../backups/backup" + dirname + "/")

def clean(dirname):
    path = os.path.dirname(os.path.realpath(__file__))
    res_list = os.listdir(path + "/../../../backups/backup" + dirname)
    if len(res_list) > 1:
        os.system("rm -f " + path + "/../results/results*")
        os.system("for d in " + path + "/params*; do for f in $d/table*; do rm $f ; done ; done")

        os.system("for add in " + path + "/add* ; do "
                  "python $add ; done")
    else:
        print("Carreful, " + dirname + " is empty. Aborting the cleaning")

if sys.argv[1] == "clean":
    dirname = str(datetime.datetime.now()).replace(" ", "")
    backup(dirname)
    clean(dirname)
elif sys.argv[1] == "backup":
    dirname = str(datetime.datetime.now()).replace(" ", "")
    backup(dirname)
else:
    print("invalid instruction")