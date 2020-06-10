import numpy as np
import pandas as pd

def compute_efficiency_ratings(statdict, HOME_CORR=3.1, conv_param=1.0, preseason_blend=0.0):
    """ Compute strength-of-schedule adjusted offensive and defensive efficiency ratings.
        Takes as input a dict of season stats as returned by utils.compute_season_stats
        HOME_CORR is an empiracle home court advantage parameter, derived when training the model
    """

    stats = ["eff","astr","tovr","efgp","orbp","ftr"]

    ## check that we've computed advanced stats. Otherwise throw error
    year = list(statdict.keys())[0]
    tid = list(statdict[year].keys())[0]
    if "Teff" not in statdict[year][tid]:
        raise Exception("Must first compute advanced stats with utils.add_advanced_stats")

    # func to compute average offensive/defensive efficiencies among all teams in a given year
    def GetAvg(year, stat):
        nteams = 0
        tot = 0.
        for tid in statdict[year]:
            if statdict[year][tid]["TFGA"] > 0:
                nteams += 1
                tot += statdict[year][tid]["Tcorr"+stat]
        return 0 if nteams==0 else tot/nteams

    # compute raw offensive/defensive efficiency for each team
    for year in statdict:
        for tid in statdict[year]:
            d = statdict[year][tid]

            for stat in stats:
                if d["TFGA"] > 0:
                    d["Tcorro"+stat] = d["T"+stat]
                    d["Tcorrd"+stat] = d["O"+stat]
                else:
                    d["Tcorro"+stat] = 0.0
                    d["Tcorrd"+stat] = 999.0

            d["pacetemp"] = d["rawpace"]
            d["pace"] = d["pacetemp"]

    # iterate until corrected efficiencies aren't changing any more
    maxchange = 100
    niter = -1
    while maxchange > 0.001:
    # for iiter in range(50):
        niter += 1
        maxchange = 0.0
        # compute opponents' average efficiencies
        for year in statdict:
            for tid in statdict[year]:
                sched = statdict[year][tid]["opps"]
                sumoff = {stat:0. for stat in stats}
                sumdef = {stat:0. for stat in stats}
                igame = 0
                norm = 0.0
                for opp,HA in zip(sched,statdict[year][tid]["HA"]):
                    igame += 1
                    norm += 1
                    for stat in stats:
                        HAmult = "ANH".find(HA) - 1 if stat=="eff" else 0
                        sumoff[stat] += (statdict[year][opp]["Tcorro"+stat] - HAmult*HOME_CORR)
                        sumdef[stat] += (statdict[year][opp]["Tcorrd"+stat] + HAmult*HOME_CORR)
                if len(sched)>0:
                    for stat in stats:
                        sumoff[stat] /= norm
                        sumdef[stat] /= norm
                        statdict[year][tid]["Ocorro"+stat] = sumoff[stat]
                        statdict[year][tid]["Ocorrd"+stat] = sumdef[stat]
                else:
                    for stat in stats:
                        statdict[year][tid]["Ocorro"+stat] = 0.0
                        statdict[year][tid]["Ocorrd"+stat] = 999.0

        # correct team efficiencies
        for year in statdict:
            avg_stat = {}
            for stat in stats:
                avg_stat["o"+stat] = GetAvg(year, "o"+stat)
                avg_stat["d"+stat] = GetAvg(year, "d"+stat)
            avg_oeff = GetAvg(year, "oeff")
            avg_deff = GetAvg(year, "deff")
            avg_pace = np.mean([statdict[year][tid]["pacetemp"] for tid in statdict[year] if statdict[year][tid]["TFGA"]>0])
            for tid in statdict[year].keys():
                d = statdict[year]
                if d[tid]["TFGA"]==0:
                    continue
                # correct offensive efficiency for avg. opponent defensive efficiency
                # (percent correction based on percent avg. opponent is better/worse than average)
                oldoeff = d[tid]["Tcorroeff"]
                olddeff = d[tid]["Tcorrdeff"]
                for stat in stats:
                    ocorr = d[tid]["Ocorrd"+stat]/avg_stat["d"+stat]
                    d[tid]["Tcorro"+stat] = d[tid]["T"+stat] / ocorr**conv_param
                    dcorr = d[tid]["Ocorro"+stat]/avg_stat["o"+stat]
                    d[tid]["Tcorrd"+stat] = d[tid]["O"+stat] / dcorr**conv_param
                maxchange = max(maxchange, 
                                max(abs(oldoeff-d[tid]["Tcorroeff"]), abs(olddeff-d[tid]["Tcorrdeff"])))
                d[tid]["pace"] = 0
                ngames = len(d[tid]["opps"])
                for i in range(ngames):
                    opp = d[tid]["opps"][i]
                    gamepace = d[tid]["poss"][i] / (1.0 + 0.125*d[tid]["nOT"][i])
                    ## comes from formula  game_pace = t1_pace * t2_pace/avg_pace
                    d[tid]["pace"] += (gamepace*(avg_pace / d[opp]["pacetemp"])**conv_param) / ngames  
                    # d[tid]["pace"] += (gamepace + avg_pace - d[opp]["pacetemp"]) / ngames  
                    # d[tid]["pace"] += (2*gamepace - d[opp]["pacetemp"]) / ngames  

        for year in statdict:
            for tid in statdict[year]:
                statdict[year][tid]["pacetemp"] = statdict[year][tid]["pace"]


    p = preseason_blend
    for year in statdict:
        for tid in statdict[year]:
            d = statdict[year][tid]
            d.pop("pacetemp")
            d["Tneteff"] = d["Tcorroeff"] - d["Tcorrdeff"]
            d["Oneteff"] = d["Ocorroeff"] - d["Ocorrdeff"]

            if d.get("preseason_eff", -999) > -900:
                if d["TFGA"] > 0:
                    d["CompositeRating"] = p*d["preseason_eff"] + (1-p)*d["Tneteff"]
                    d["CompositeOff"] = p*d["preseason_oeff"] + (1-p)*d["Tcorroeff"]
                    d["CompositeDef"] = p*d["preseason_deff"] + (1-p)*d["Tcorrdeff"]
                    d["CompositePace"] = p*d["preseason_pace"] + (1-p)*d["pace"]
                else:
                    d["CompositeRating"] = d["preseason_eff"]
                    d["CompositeOff"] = d["preseason_oeff"]
                    d["CompositeDef"] = d["preseason_deff"]
                    d["CompositePace"] = d["preseason_pace"]
            else:
                d["CompositeRating"] = d["Tneteff"]
                d["CompositeOff"] = d["Tcorroeff"]
                d["CompositeDef"] = d["Tcorrdeff"]
                d["CompositePace"] = d["pace"]
                
                    



