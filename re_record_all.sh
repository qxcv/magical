#!/bin/bash

set -e

# cluster-colour-2020-05-18
# cluster-shape-2020-05-18
# find-dupe-2020-05-18
# fix-colour-2020-05-22
# make-line-2020-05-22
# match-regions-2020-05-18
# move-to-corner-2020-03-01
# move-to-region-2020-04-04

xvfb-run -a python -m milbench.misc.re_record_demos ClusterColour-Demo-v0 \
         ./demos-simplified/cluster-colour-2020-05-18/*.pkl.gz \
         --out-dir demos-ea/cluster-colour-2020-05-18/

xvfb-run -a python -m milbench.misc.re_record_demos ClusterShape-Demo-v0 \
         ./demos-simplified/cluster-shape-2020-05-18/*.pkl.gz \
         --out-dir demos-ea/cluster-shape-2020-05-18/

xvfb-run -a python -m milbench.misc.re_record_demos FindDupe-Demo-v0 \
         ./demos-simplified/find-dupe-2020-05-18/*.pkl.gz \
         --out-dir demos-ea/find-dupe-2020-05-18/

xvfb-run -a python -m milbench.misc.re_record_demos FixColour-Demo-v0 \
         ./demos-simplified/fix-colour-2020-05-22/*.pkl.gz \
         --out-dir demos-ea/fix-colour-2020-05-22/

xvfb-run -a python -m milbench.misc.re_record_demos MakeLine-Demo-v0 \
         ./demos-simplified/make-line-2020-05-22/*.pkl.gz \
         --out-dir demos-ea/make-line-2020-05-22/

xvfb-run -a python -m milbench.misc.re_record_demos MatchRegions-Demo-v0 \
         ./demos-simplified/match-regions-2020-05-18/*.pkl.gz \
         --out-dir demos-ea/match-regions-2020-05-18/

xvfb-run -a python -m milbench.misc.re_record_demos MoveToCorner-Demo-v0 \
         ./demos-simplified/move-to-corner-2020-03-01/*.pkl.gz \
         --out-dir demos-ea/move-to-corner-2020-03-01/

xvfb-run -a python -m milbench.misc.re_record_demos MoveToRegion-Demo-v0 \
         ./demos-simplified/move-to-region-2020-04-04/*.pkl.gz \
         --out-dir demos-ea/move-to-region-2020-04-04/
