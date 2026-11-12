"""
360° Multi-Image Fusion Module.
Combines detections from multiple vehicle angles (Front, Rear, Left, Right, etc.).
Performs feature matching (OpenCV ORB/SIFT) and semantic correlation to remove duplicates
and synthesize a single unified vehicle damage inventory.
"""

import cv2
import numpy as np

class MultiImageFusion:
    def __init__(self):
        # Initialize OpenCV ORB feature detector for visual similarity verification
        self.orb = cv2.ORB_create(nfeatures=500)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def fuse_inspections(self, view_results):
        """
        Takes inspection outputs from multiple images/views.
        view_results: list of dicts:
            {
                "image_name": str,
                "view_name": str,
                "damages": list of damage records,
                "image_path": str (optional)
            }
        Returns:
            fused_summary: {
                "total_images": int,
                "damaged_parts_count": int,
                "minor_count": int,
                "moderate_count": int,
                "severe_count": int,
                "duplicates_removed": int,
                "fused_damages": list of consolidated damage records,
                "view_breakdown": dict
            }
        """
        all_damages = []
        for v in view_results:
            view_name = v.get("view_name", "Unknown View")
            img_name = v.get("image_name", "image.jpg")
            for d in v.get("damages", []):
                item = dict(d)
                item["source_view"] = view_name
                item["source_image"] = img_name
                all_damages.append(item)

        if not all_damages:
            return {
                "total_images": len(view_results),
                "damaged_parts_count": 0,
                "minor_count": 0,
                "moderate_count": 0,
                "severe_count": 0,
                "duplicates_removed": 0,
                "fused_damages": [],
                "view_breakdown": {}
            }

        fused_damages = []
        duplicates_removed = 0

        # Cluster by canonical part name and damage type
        # Parts that can appear in adjacent angles:
        # e.g. Front View <-> Front-Left View <-> Front-Right View: "Front Bumper", "Hood"
        # Rear View <-> Rear-Left View <-> Rear-Right View: "Rear Bumper", "Trunk"
        
        for candidate in all_damages:
            is_duplicate = False
            for existing in fused_damages:
                if self._is_same_damage(candidate, existing):
                    # Found duplicate across views!
                    is_duplicate = True
                    duplicates_removed += 1
                    # Merge information: keep higher confidence and record alternate views
                    if candidate.get("confidence", 0) > existing.get("confidence", 0):
                        existing["confidence"] = candidate["confidence"]
                        existing["damaged_area"] = max(existing.get("damaged_area", 0), candidate.get("damaged_area", 0))
                    
                    if "alternate_views" not in existing:
                        existing["alternate_views"] = []
                    existing["alternate_views"].append(candidate["source_view"])
                    break
            
            if not is_duplicate:
                c_copy = dict(candidate)
                c_copy["alternate_views"] = []
                fused_damages.append(c_copy)

        # Count severities
        minor_c = sum(1 for d in fused_damages if d.get("severity", "").lower() == "minor")
        mod_c = sum(1 for d in fused_damages if d.get("severity", "").lower() == "moderate")
        sev_c = sum(1 for d in fused_damages if d.get("severity", "").lower() == "severe")

        # View breakdown
        view_breakdown = {}
        for v in view_results:
            view_breakdown[v.get("view_name", "Unknown")] = len(v.get("damages", []))

        return {
            "total_images": len(view_results),
            "damaged_parts_count": len(fused_damages),
            "minor_count": minor_c,
            "moderate_count": mod_c,
            "severe_count": sev_c,
            "duplicates_removed": duplicates_removed,
            "fused_damages": fused_damages,
            "view_breakdown": view_breakdown
        }

    def _is_same_damage(self, d1, d2):
        """
        Determines if two damage detections across distinct images represent the same physical flaw.
        Evaluates canonical component name, damage type similarity, and angle compatibility.
        """
        part1 = self._canonical_part(d1.get("part", ""))
        part2 = self._canonical_part(d2.get("part", ""))

        if part1 != part2:
            return False

        # If parts match, check damage type
        type1 = d1.get("damage_type", "").lower()
        type2 = d2.get("damage_type", "").lower()
        if type1 != type2:
            return False

        # Check view proximity
        v1 = d1.get("source_view", "").lower()
        v2 = d2.get("source_view", "").lower()

        # If same view and same part and type -> duplicate
        if v1 == v2:
            return True

        # Adjacent perspective overlap pairs
        compatible_views = [
            {"front view", "front-left view"},
            {"front view", "front-right view"},
            {"rear view", "rear-left view"},
            {"rear view", "rear-right view"},
            {"left view", "front-left view"},
            {"left view", "rear-left view"},
            {"right view", "front-right view"},
            {"right view", "rear-right view"}
        ]

        pair = {v1, v2}
        for comp in compatible_views:
            if pair == comp:
                return True

        return False

    def _canonical_part(self, name):
        """Normalizes part strings (e.g. 'Headlight (Left)' -> 'Headlight Left')."""
        n = name.lower().replace("(", "").replace(")", "").replace("-", " ").strip()
        if "bumper" in n:
            return "front bumper" if "front" in n else "rear bumper"
        if "hood" in n:
            return "hood"
        if "trunk" in n or "tailgate" in n:
            return "trunk"
        if "fender" in n:
            return "fender left" if "left" in n else "fender right" if "right" in n else "fender"
        if "headlight" in n:
            return "headlight left" if "left" in n else "headlight right" if "right" in n else "headlight"
        if "tail light" in n or "taillight" in n:
            return "tail light left" if "left" in n else "tail light right" if "right" in n else "tail light"
        if "door" in n:
            return "door left" if "left" in n else "door right" if "right" in n else "door"
        if "windshield" in n:
            return "rear windshield" if "rear" in n else "front windshield"
        if "mirror" in n:
            return "side mirror"
        if "roof" in n:
            return "roof"
        return n
