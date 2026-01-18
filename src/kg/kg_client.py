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
}


class KGClient:

    """Client for querying the RDF knowledge graph.

    This client loads a Turtle (.ttl) knowledge graph and provides helpers to
    retrieve:
    - maintenance procedures that mitigate a BN-inferred cause
    - context for a cause (affected components and related symptoms)
    - explanations/details for a specific procedure
    """

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
        """Return all procedures that mitigate a given BN cause.

        Args:
            bn_cause: BN cause variable name (e.g., ``"fan_fault"``).

        Returns (str):
            A list of procedure dicts with keys:
                - proc_uri
                - proc_id
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
                "proc_id": str(row.proc).split("#")[-1],
                "proc_name": str(row.proc_name),
                "effort_h": float(row.effort),
                "spare_parts_cost_eur": float(row.cost),
                "risk_rating": int(row.risk),
                "target_component": str(row.comp_name),
            })
        return procedures

    def get_cause_context(self, bn_cause: str) -> dict[str, Any]:
        """Get context for a BN cause (affected components and symptoms).

        Args:
            bn_cause: BN cause variable name (e.g., ``"fan_fault"``).

        Returns:
            A dictionary with keys:
                - bn_cause
                - cause_name (KG local name)
                - components (sorted list)
                - symptoms (sorted list)
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
        """Get an explanation/details record for a procedure.

        Args:
            proc_id: Procedure ID / KG local name (e.g., ``"P1"``).

        Returns:
            A dictionary with keys:
                - proc_id
                - proc_name
                - target_component
                - cause_name
                - effort_h
                - spare_parts_cost_eur
                - symptoms (sorted list)
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
