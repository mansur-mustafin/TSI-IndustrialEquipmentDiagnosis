# kg_client.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

from rdflib import Graph, Namespace


EX = Namespace("http://example.org/cnc#")

BN_TO_CAUSE_NAME = {
    "bearing_wear_high": "BearingWearHigh",
    "fan_fault": "FanFault",
    "clogged_filter": "CloggedFilter",
    "low_cooling_efficiency": "LowCoolingEfficiency",
}


@dataclass
class KGClientConfig:
    ttl_path: str


class KGClient:
    """
    Small client to query the Knowledge Graph built with rdflib.

    Main methods you will use from the BN side:
      - get_procedures_for_bn_cause(bn_cause)
      - get_cause_context(bn_cause)
    """

    def __init__(self, cfg: KGClientConfig):
        self.cfg = cfg
        self.graph = Graph()
        self.graph.parse(cfg.ttl_path, format="turtle")
        self.graph.bind("ex", EX)

    def _bn_cause_to_kg_name(self, bn_cause: str) -> str:
        """
        Map BN cause variable name to KG cause name used in IRI.
          'fan_fault' -> 'FanFault'
        """
        try:
            return BN_TO_CAUSE_NAME[bn_cause]
        except KeyError:
            raise ValueError(f"Unknown BN cause name: {bn_cause!r}")

    def get_procedures_for_bn_cause(self, bn_cause: str) -> List[Dict[str, Any]]:
        """
        Given a BN cause variable (e.g. 'fan_fault'),
        return all procedures that mitigate that cause.

        Output: list of dicts with keys:
          - proc_uri
          - proc_name
          - effort_h
          - spare_parts_cost_eur
          - risk_rating
          - target_component
        """
        cause_local = self._bn_cause_to_kg_name(bn_cause)

        query = f"""
        PREFIX ex: <http://example.org/cnc#>
        SELECT ?proc ?proc_name ?effort ?cost ?risk ?comp_name
        WHERE {{
          ?proc a ex:Procedure ;
                ex:mitigatesCause ex:{cause_local} ;
                ex:hasName ?proc_name ;
                ex:effortHours ?effort ;
                ex:sparePartsCostEur ?cost ;
                ex:riskRating ?risk ;
                ex:targetsComponent ?comp .
          ?comp ex:hasName ?comp_name .
        }}
        """

        results = self.graph.query(query)

        procedures = []
        for row in results:
            procedures.append({
                "proc_uri": str(row.proc),
                "proc_name": str(row.proc_name),
                "effort_h": float(row.effort),
                "spare_parts_cost_eur": float(row.cost),
                "risk_rating": int(row.risk),
                "target_component": str(row.comp_name),
            })
        return procedures

    def get_cause_context(self, bn_cause: str) -> Dict[str, Any]:
        """
        Given BN cause name, return:
          - cause_name  (KG name)
          - components: list of component names affected by this cause
          - symptoms:   list of symptom names this cause can produce

        This is useful for explanations like:
          "FanFault affects component Fan and causes symptom HighSpindleTemp."
        """
        cause_local = self._bn_cause_to_kg_name(bn_cause)

        query = f"""
        PREFIX ex: <http://example.org/cnc#>
        SELECT ?comp_name ?sym_name
        WHERE {{
          OPTIONAL {{
            ex:{cause_local} ex:affectsComponent ?comp .
            ?comp ex:hasName ?comp_name .
          }}
          OPTIONAL {{
            ex:{cause_local} ex:causesSymptom ?sym .
            ?sym ex:hasName ?sym_name .
          }}
        }}
        """

        results = self.graph.query(query)

        components = set()
        symptoms = set()
        for row in results:
            if row.comp_name:
                components.add(str(row.comp_name))
            if row.sym_name:
                symptoms.add(str(row.sym_name))

        return {
            "bn_cause": bn_cause,
            "cause_name": cause_local,
            "components": sorted(components),
            "symptoms": sorted(symptoms),
        }

    def get_procedure_explanation(self, proc_id: str) -> Dict[str, Any]:
        """
        Given a procedure ID (e.g. 'P1'), get:
          - procedure name
          - target component
          - mitigated cause
          - cause's symptoms

        Useful when you want to generate text like:
          "ReplaceBearing targets Bearing and mitigates BearingWearHigh,
           which is associated with symptoms [HighVibration]."
        """
        query = f"""
        PREFIX ex: <http://example.org/cnc#>
        SELECT ?proc_name ?comp_name ?cause_name ?sym_name
        WHERE {{
          ex:{proc_id} a ex:Procedure ;
                        ex:hasName ?proc_name ;
                        ex:targetsComponent ?comp ;
                        ex:mitigatesCause ?cause .
          ?comp ex:hasName ?comp_name .
          ?cause ex:hasName ?cause_name .
          OPTIONAL {{
            ?cause ex:causesSymptom ?sym .
            ?sym ex:hasName ?sym_name .
          }}
        }}
        """

        results = self.graph.query(query)

        proc_name = None
        comp_name = None
        cause_name = None
        symptoms = set()

        for row in results:
            proc_name = str(row.proc_name)
            comp_name = str(row.comp_name)
            cause_name = str(row.cause_name)
            if row.sym_name:
                symptoms.add(str(row.sym_name))

        if proc_name is None:
            raise ValueError(f"No procedure with id {proc_id!r} found in KG.")

        return {
            "proc_id": proc_id,
            "proc_name": proc_name,
            "target_component": comp_name,
            "cause_name": cause_name,
            "symptoms": sorted(symptoms),
        }
