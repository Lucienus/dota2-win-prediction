"""Versioned tower-event correction; not a replacement for frozen v2 weights.

This adapter is a tested prerequisite for the planned retraining experiments.
It does not assert that the correction improves prediction accuracy.
"""
import re
from data_protocol import finite_number
from feature_extractor import MatchFeatureExtractor

TOWER_KEY=re.compile(r'^npc_dota_(goodguys|badguys)_tower([1-4])(?:_(top|mid|bot))?$')

def tower_events_before(match,cutoff):
    log=match.get('objectives')
    if not isinstance(log,list):raise ValueError('Missing objective log cannot mean zero towers')
    towers=[]
    for event in log:
        if not isinstance(event,dict) or not finite_number(event.get('time')):raise ValueError('Malformed objective timestamp')
        if event['time']>cutoff:continue
        if str(event.get('type','')).lower()!='building_kill':continue
        key=event.get('key','')
        if not isinstance(key,str):raise ValueError('Malformed building key')
        parsed=TOWER_KEY.fullmatch(key)
        if parsed is None:
            if 'tower' in key:raise ValueError('Unrecognized tower key: '+key)
            continue
        side,tier,lane=parsed.groups()
        if tier!='4' and lane is None:raise ValueError('Missing tower lane')
        if tier=='4' and lane=='mid':raise ValueError('Unrecognized tier4 mid tower')
        # Two distinct tier4 towers can share the same key. Keys are type names,
        # not entity IDs; deduplicating by key would undercount their destruction.
        towers.append({'time':event['time'],'owner':'radiant' if side=='goodguys' else 'dire','tier':int(tier),'lane':lane,'key':key})
    return sorted(towers,key=lambda e:(e['time'],e['key']))

def tower_counts_before(match,cutoff):
    events=tower_events_before(match,cutoff)
    # Count opposing structures lost. This does not identify the last-hit player.
    return sum(e['owner']=='dire' for e in events),sum(e['owner']=='radiant' for e in events)

class TowerCorrectedExtractor(MatchFeatureExtractor):
    """Same 20 numerical node columns and targets, fixing only columns15/16."""
    def extract_inputs(self):
        result=super().extract_inputs()
        radiant,dire=tower_counts_before(self.data,self.target_sec)
        result['nodes'][:,15]=radiant/11
        result['nodes'][:,16]=dire/11
        return result
