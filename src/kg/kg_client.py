from typing import Any

from kg.kg_client_config import KGClientConfig
from rdflib import Graph, Namespace

EX = Namespace("http://example.org/cnc#")

BN_TO_CAUSE_NAME = {
    "bearing_wear_high": "BearingWearHigh",
    "fan_fault": "FanFault",
    "clogged_filter": "CloggedFilter",
    "low_cooling_efficiency": "LowCoolingEfficiency",
    "low_load_condition": "LowLoadCondition",
    "low_cooling_effectiveness": "LowCoolingEffectiveness",
}


class KGClient:

    def __init__(self, cfg: KGClientConfig):
        self.graph = Graph()
        self.graph.parse(cfg.ttl_path, format="turtle")
        self.graph.bind("ex", EX)

    def _bn_cause_to_kg_name(self, bn_cause: str) -> str:
        try:
            return BN_TO_CAUSE_NAME[bn_cause]
        except KeyError:
            raise ValueError(f"Unknown BN cause name: {bn_cause!r}")

    def get_procedures_for_cause(self, bn_cause: str) -> list[dict[str, Any]]:
        """
        Given a BN cause variable returns all procedures that mitigate that cause.

        :param bn_cause: A BN cause variable name (e.g. 'fan_fault')
        :type bn_cause: str
        :return: list of procedures
        :rtype: list[dict[str, Any]]
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
                "proc_id": str(row.proc).split("#")[-1],
                "proc_name": str(row.proc_name),
                "effort_h": float(row.effort),
                "spare_parts_cost_eur": float(row.cost),
                "risk_rating": int(row.risk),
                "target_component": str(row.comp_name),
            })
        return procedures

    def get_cause_context(self, bn_cause: str) -> dict[str, Any]:
        """
        Given a BN cause variable, get its context:
        
        :param bn_cause: A BN cause variable name (e.g. 'fan_fault')
        :type bn_cause: str
        :return: Dictionary with cause context
        :rtype: dict[str, Any]
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

    def get_procedure_explanation(self, proc_id: str) -> dict[str, Any]:
        """
        Given a procedure ID, get its explanation.

        :param proc_id: Procedure ID (e.g. 'P1')
        :type proc_id: str
        :return: Dictionary with procedure explanation
        :rtype: dict[str, Any]
        """
        query = f"""
        PREFIX ex: <http://example.org/cnc#>
        SELECT ?proc_name ?comp_name ?cause_name ?sym_name ?effort ?cost
        WHERE {{
            ex:{proc_id} a ex:Procedure ;
                            ex:hasName ?proc_name ;
                            ex:targetsComponent ?comp ;
                            ex:mitigatesCause ?cause ;
                            ex:effortHours ?effort ;
                            ex:sparePartsCostEur ?cost .
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
        effort_h = None
        parts_cost = None
        symptoms = set()

        for row in results:
            proc_name = str(row.proc_name)
            comp_name = str(row.comp_name)
            cause_name = str(row.cause_name)
            effort_h = float(row.effort)
            parts_cost = float(row.cost)
            if row.sym_name:
                symptoms.add(str(row.sym_name))

        if proc_name is None:
            raise ValueError(f"No procedure with id {proc_id!r} found in KG.")

        return {
            "proc_id": proc_id,
            "proc_name": proc_name,
            "target_component": comp_name,
            "cause_name": cause_name,
            "effort_h": effort_h,
            "spare_parts_cost_eur": parts_cost,
            "symptoms": sorted(symptoms),
        }
