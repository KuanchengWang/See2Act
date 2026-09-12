# Copyright 2023 The Ravens Authors. Licensed under the Apache License, Version 2.0.
"""Suction gripper (subset of ravens.tasks.grippers)."""
import os

import numpy as np
import pybullet as p

from see2act.sim.pybullet_utils import load_urdf

SUCTION_BASE_URDF = "ur5/suction/suction-base.urdf"
SUCTION_HEAD_URDF = "ur5/suction/suction-head.urdf"


class Suction:
    """Simple suction dynamics: a fixed constraint to the contacted rigid object."""

    def __init__(self, assets_root, robot, ee, obj_ids):
        self.assets_root = assets_root
        pose = ((0.487, 0.109, 0.438), p.getQuaternionFromEuler((np.pi, 0, 0)))
        base = load_urdf(p, os.path.join(self.assets_root, SUCTION_BASE_URDF), pose[0], pose[1])
        p.createConstraint(parentBodyUniqueId=robot, parentLinkIndex=ee, childBodyUniqueId=base, childLinkIndex=-1,
                           jointType=p.JOINT_FIXED, jointAxis=(0, 0, 0), parentFramePosition=(0, 0, 0),
                           childFramePosition=(0, 0, 0.01))

        pose = ((0.487, 0.109, 0.347), p.getQuaternionFromEuler((np.pi, 0, 0)))
        self.body = load_urdf(p, os.path.join(self.assets_root, SUCTION_HEAD_URDF), pose[0], pose[1])
        constraint_id = p.createConstraint(parentBodyUniqueId=robot, parentLinkIndex=ee, childBodyUniqueId=self.body,
                                           childLinkIndex=-1, jointType=p.JOINT_FIXED, jointAxis=(0, 0, 0),
                                           parentFramePosition=(0, 0, 0), childFramePosition=(0, 0, -0.08))
        p.changeConstraint(constraint_id, maxForce=50)

        self.obj_ids = obj_ids
        self.activated = False
        self.contact_constraint = None

    def activate(self):
        """Attach the contacted rigid object with a fixed constraint."""
        if not self.activated:
            points = p.getContactPoints(bodyA=self.body, linkIndexA=0)
            if points:
                for point in points:
                    obj_id, contact_link = point[2], point[4]
                if obj_id in self.obj_ids["rigid"]:
                    body_pose = p.getLinkState(self.body, 0)
                    obj_pose = p.getBasePositionAndOrientation(obj_id)
                    world_to_body = p.invertTransform(body_pose[0], body_pose[1])
                    obj_to_body = p.multiplyTransforms(world_to_body[0], world_to_body[1], obj_pose[0], obj_pose[1])
                    self.contact_constraint = p.createConstraint(
                        parentBodyUniqueId=self.body, parentLinkIndex=0, childBodyUniqueId=obj_id,
                        childLinkIndex=contact_link, jointType=p.JOINT_FIXED, jointAxis=(0, 0, 0),
                        parentFramePosition=obj_to_body[0], parentFrameOrientation=obj_to_body[1],
                        childFramePosition=(0, 0, 0), childFrameOrientation=(0, 0, 0))
                self.activated = True

    def release(self):
        if self.activated:
            self.activated = False
            if self.contact_constraint is not None:
                try:
                    p.removeConstraint(self.contact_constraint)
                    self.contact_constraint = None
                except Exception:  # pragma: no cover
                    pass

    def detect_contact(self):
        """Contact between the suction head (or the held object) and any other body."""
        body, link = self.body, 0
        if self.activated and self.contact_constraint is not None:
            try:
                info = p.getConstraintInfo(self.contact_constraint)
                body, link = info[2], info[3]
            except Exception:  # pragma: no cover
                self.contact_constraint = None
        points = p.getContactPoints(bodyA=body, linkIndexA=link)
        if self.activated:
            points = [point for point in points if point[2] != self.body]
        return len(points) > 0

    def check_grasp(self):
        return self.contact_constraint is not None
