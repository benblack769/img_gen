import os
import subprocess

dir = "/home/ben/Downloads/img_net/"
tars = os.listdir(dir)
for tar in tars:
    outname = tar.replace(".tar","")
    outdir = dir+outname+"/"
    os.mkdir(outdir)
    subprocess.check_call(['tar',"xvf",dir+tar],cwd=outdir)
