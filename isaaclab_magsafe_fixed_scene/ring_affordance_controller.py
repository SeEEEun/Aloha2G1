"""Geometry-derived, ring-relative right-hand controller for MagSafe removal.

All stored targets are expressed in the instantaneous ring frame.  World poses
exist for one physics step only and are never used as calibrated waypoints.
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np


class Stage(str, Enum):
    RECORDED_APPROACH = "RECORDED_APPROACH"
    PREINSERTION_BLEND = "PREINSERTION_BLEND"
    RING_RELATIVE_INSERTION = "RING_RELATIVE_INSERTION"
    PHYSICAL_GRASP = "PHYSICAL_GRASP"
    PHYSICAL_PULL = "PHYSICAL_PULL"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


@dataclass
class RingFrame:
    origin: np.ndarray
    rotation: np.ndarray  # columns are ring +X,+Y,+Z in world

    def world(self, p_ring: np.ndarray) -> np.ndarray:
        return self.origin + self.rotation @ np.asarray(p_ring)

    def local(self, p_world: np.ndarray) -> np.ndarray:
        return self.rotation.T @ (np.asarray(p_world) - self.origin)


@dataclass
class FingerState:
    center: np.ndarray
    quaternion_xyzw: np.ndarray
    longitudinal: np.ndarray
    wide: np.ndarray
    thin: np.ndarray
    opposing_center: np.ndarray


@dataclass
class ContactState:
    insertion_inner: bool = False
    opposing_outer: bool = False
    side_collision: bool = False
    phone_collision: bool = False
    max_penetration_m: float = 0.0
    pull_force_n: float = 0.0


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float)
    return v / max(float(np.linalg.norm(v)), 1.0e-12)


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(q, float)
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


def matrix_to_quat(r: np.ndarray) -> np.ndarray:
    r = np.asarray(r, float)
    q = np.empty(4)
    tr = float(np.trace(r))
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q[:] = [(r[2,1]-r[1,2])/s, (r[0,2]-r[2,0])/s,
                (r[1,0]-r[0,1])/s, 0.25*s]
    else:
        i = int(np.argmax(np.diag(r)))
        if i == 0:
            s = math.sqrt(1+r[0,0]-r[1,1]-r[2,2])*2
            q[:] = [0.25*s, (r[0,1]+r[1,0])/s,
                    (r[0,2]+r[2,0])/s, (r[2,1]-r[1,2])/s]
        elif i == 1:
            s = math.sqrt(1+r[1,1]-r[0,0]-r[2,2])*2
            q[:] = [(r[0,1]+r[1,0])/s, 0.25*s,
                    (r[1,2]+r[2,1])/s, (r[0,2]-r[2,0])/s]
        else:
            s = math.sqrt(1+r[2,2]-r[0,0]-r[1,1])*2
            q[:] = [(r[0,2]+r[2,0])/s, (r[1,2]+r[2,1])/s,
                    0.25*s, (r[1,0]-r[0,1])/s]
    return q / np.linalg.norm(q)


def rotation_vector(r: np.ndarray) -> np.ndarray:
    c = float(np.clip((np.trace(r)-1)*0.5, -1, 1))
    angle = math.acos(c)
    if angle < 1e-8:
        return 0.5*np.array([r[2,1]-r[1,2], r[0,2]-r[2,0], r[1,0]-r[0,1]])
    return angle/(2*math.sin(angle))*np.array(
        [r[2,1]-r[1,2], r[0,2]-r[2,0], r[1,0]-r[0,1]]
    )


def minimum_jerk(u: float) -> float:
    u = float(np.clip(u, 0, 1))
    return 10*u**3 - 15*u**4 + 6*u**5


class RingPoseProvider:
    """Interface returning the current affordance frame."""
    name = "abstract"

    def get(self, accessory_pose_xyzw: np.ndarray) -> RingFrame:
        raise NotImplementedError


class SimulationGroundTruthRingPoseProvider(RingPoseProvider):
    name = "simulation_ground_truth"

    def get(self, accessory_pose_xyzw: np.ndarray) -> RingFrame:
        pose = np.asarray(accessory_pose_xyzw, float)
        ra = quat_to_matrix(pose[3:7])
        # A: +X=accessory +X, +Y=accessory -Z, +Z=accessory +Y.
        return RingFrame(pose[:3].copy(), ra @ np.array([
            [1., 0., 0.],
            [0., 0., 1.],
            [0.,-1., 0.],
        ]))


class FixedCalibrationRingPoseProvider(SimulationGroundTruthRingPoseProvider):
    name = "fixed_calibration"

    def __init__(self):
        self._frame = None

    def get(self, accessory_pose_xyzw: np.ndarray) -> RingFrame:
        if self._frame is None:
            self._frame = super().get(accessory_pose_xyzw)
        return self._frame


class PerceptionEstimateRingPoseProvider(SimulationGroundTruthRingPoseProvider):
    name = "perception_estimate"


class RingAffordanceController:
    INNER_RADIUS = 0.0225
    OUTER_RADIUS = 0.0275
    AXIAL_THICKNESS = 0.0035
    CROSS_SECTION = np.array([0.004979, 0.019656])  # thin, wide
    CLEARANCE_REQUIRED = 0.001
    INSERTION_FINGER = "follower_right_carriage_left"
    OPPOSING_FINGER = "follower_right_carriage_right"

    def __init__(self, report_root: Path, dt: float, provider: str):
        self.root = Path(report_root)
        self.root.mkdir(parents=True, exist_ok=True)
        providers = {
            "simulation_ground_truth": SimulationGroundTruthRingPoseProvider,
            "perception_estimate": PerceptionEstimateRingPoseProvider,
            "fixed_calibration": FixedCalibrationRingPoseProvider,
        }
        self.provider = providers[provider]()
        self.dt = float(dt)
        self.stage = Stage.RECORDED_APPROACH
        self.stage_time = 0.0
        self.blend_duration = 0.0
        self.blend_start_pose = None
        self.target_rotation = None
        self.body_to_finger_frame = None
        self.pre_distance = 0.0
        self.insertion_depth = 0.0
        self.grasp_hold = 0.0
        self.pull_force_hold = 0.0
        self.pull_distance = 0.0
        self.fail_reason = ""
        self.close_enabled = False
        self.plane_crossed = False
        self.side_collision = False
        self.bilateral = False
        self.max_penetration = 0.0
        self.max_pull_force = 0.0
        self.detach_frame = None
        self.target_count = self.ik_success = 0
        self.max_pos_error = self.max_rot_error = self.max_joint_step = 0.0
        self.max_joint_velocity = self.max_joint_acceleration = 0.0
        self.prev_q = self.prev_dq = None
        self.max_linear_velocity = 0.20
        self.max_angular_velocity = math.radians(120)
        self.max_linear_acceleration = 1.0
        self.max_angular_acceleration = math.radians(600)
        self.last_target_pose = None
        self.last_target_velocity = np.zeros(6)
        self.recorded_translation_correction = np.zeros(3)
        self.recorded_orientation_correction_deg = 0.0
        self.minimum_clearance = -float("inf")
        self.previous_ring_distance = None
        self._open_outputs()
        self._write_config()

    def _exclusive(self, name: str, newline=None):
        return (self.root/name).open("x", encoding="utf-8", newline=newline)

    def _open_outputs(self):
        self.frame_file = self._exclusive("ring_frame_runtime_timeline.csv", "")
        self.frame_writer = csv.writer(self.frame_file)
        self.frame_writer.writerow(["sim_time","source_frame","stage","origin_x","origin_y","origin_z",
                                    "x_x","x_y","x_z","y_x","y_y","y_z","z_x","z_y","z_z"])
        self.target_file = self._exclusive("ring_relative_cartesian_targets.csv", "")
        self.target_writer = csv.writer(self.target_file)
        self.target_writer.writerow(["sim_time","source_frame","stage","x_A","y_A","z_A",
                                     "qx_A","qy_A","qz_A","qw_A","clearance_m","ik_success",
                                     "position_error_m","orientation_error_deg"])
        self.contact_file = self._exclusive("ring_insertion_contact_timeline.csv", "")
        self.contact_writer = csv.writer(self.contact_file)
        self.contact_writer.writerow(["sim_time","source_frame","stage","insertion_inside",
                                      "plane_crossed","clearance_m","insertion_inner_contact",
                                      "opposing_outer_contact","side_collision","phone_collision",
                                      "penetration_m","pull_force_n","close_enabled"])
        self.q_log, self.pose_log, self.stage_log = [], [], []

    def _write_config(self):
        config = {
            "ring_pose_provider": self.provider.name,
            "ring_frame": {"origin":"opening geometric center","x":"accessory +X",
                           "y":"accessory -Z","z":"phone rear outward normal"},
            "inner_radius_m": self.INNER_RADIUS, "outer_radius_m": self.OUTER_RADIUS,
            "axial_thickness_m": self.AXIAL_THICKNESS,
            "insertion_finger": self.INSERTION_FINGER,
            "opposing_finger": self.OPPOSING_FINGER,
            "finger_cross_section_thin_wide_m": self.CROSS_SECTION.tolist(),
            "minimum_clearance_m": self.CLEARANCE_REQUIRED,
            "world_waypoints": False, "finger_teleport": False,
            "detach_gate": "bilateral_contact_and_outward_force_and_hold_time",
        }
        with self._exclusive("ring_affordance_controller_config.json") as f:
            json.dump(config, f, indent=2)

    def _initialize_geometry(self, ring: RingFrame, finger: FingerState):
        # Preserve the exact body-to-measured-finger triad, but command the full
        # triad: wide=A.X, thin=A.Y, longitudinal=-A.Z (inward).
        measured = np.column_stack((_unit(finger.wide), _unit(finger.thin),
                                    _unit(finger.longitudinal)))
        normal_sign = 1.0 if np.dot(finger.longitudinal, ring.rotation[:,2]) >= 0 else -1.0
        desired_longitudinal = normal_sign * ring.rotation[:,2]
        # The opening is circular, so spin about the normal does not change
        # clearance.  Select the projected current wide axis to minimize the
        # SO(3) correction while still constraining all three axes.
        desired_wide = finger.wide - np.dot(
            finger.wide, desired_longitudinal
        ) * desired_longitudinal
        if np.linalg.norm(desired_wide) < 1.0e-6:
            desired_wide = ring.rotation[:,0]
        desired_wide = _unit(desired_wide)
        desired_thin = np.cross(desired_longitudinal, desired_wide)
        desired = np.column_stack((desired_wide, desired_thin, desired_longitudinal))
        current_body = quat_to_matrix(finger.quaternion_xyzw)
        self.body_to_finger_frame = current_body.T @ measured
        self.target_rotation = desired @ self.body_to_finger_frame.T
        half_diagonal = 0.5*float(np.linalg.norm(self.CROSS_SECTION))
        sleeve_half_length = 0.010
        self.pre_distance = self.AXIAL_THICKNESS/2 + sleeve_half_length + 0.002
        # The full distal collision envelope must remain outside the ring while
        # the orientation changes; include sleeve length and in-plane radius.
        self.pre_distance += sleeve_half_length + half_diagonal
        self.insertion_depth = self.AXIAL_THICKNESS + sleeve_half_length + 0.002
        self.minimum_clearance = self.INNER_RADIUS - half_diagonal

    def geometry(self, ring: RingFrame, finger: FingerState):
        p = ring.local(finger.center)
        # Exact rectangle vertices projected to the ring plane.
        corners = []
        for sw in (-1,1):
            for st in (-1,1):
                world = finger.center + sw*0.5*self.CROSS_SECTION[1]*finger.wide \
                    + st*0.5*self.CROSS_SECTION[0]*finger.thin
                corners.append(ring.local(world)[:2])
        radii = np.linalg.norm(np.asarray(corners), axis=1)
        clearance = self.INNER_RADIUS - float(np.max(radii))
        opposing = ring.local(finger.opposing_center)
        return {
            "center_A": p, "polygon_A_xy": np.asarray(corners),
            "clearance": clearance,
            "inside": clearance > self.CLEARANCE_REQUIRED,
            "crossed": p[2] < -self.AXIAL_THICKNESS/2,
            "opposing_outside": np.linalg.norm(opposing[:2]) > self.INNER_RADIUS,
        }

    def _pose_target(self, ring: RingFrame, z_A: float):
        p_A = np.array([0., 0., z_A])
        return ring.world(p_A), self.target_rotation, p_A

    def _smooth_pose(self, p0, r0, p1, r1, u):
        s = minimum_jerk(u)
        rv = rotation_vector(r1 @ r0.T)
        angle = np.linalg.norm(rv)
        if angle < 1e-10:
            r = r0
        else:
            k = rv/angle
            kx = np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]])
            r = (np.eye(3)+math.sin(s*angle)*kx+(1-math.cos(s*angle))*(kx@kx))@r0
        return p0+s*(p1-p0), r

    def compute_target(self, sim_time: float, frame: int, accessory_pose: np.ndarray,
                       finger: FingerState, contact: ContactState, recorded_close: bool):
        ring = self.provider.get(accessory_pose)
        if self.target_rotation is None:
            self._initialize_geometry(ring, finger)
        geo = self.geometry(ring, finger)
        self.minimum_clearance = min(self.minimum_clearance, geo["clearance"])
        self.max_penetration = max(self.max_penetration, contact.max_penetration_m)
        self.max_pull_force = max(self.max_pull_force, contact.pull_force_n)
        if contact.side_collision and not self.plane_crossed:
            self.side_collision = True
            self.stage, self.fail_reason = Stage.FAILED, "RING_INSERTION_FAIL_SIDE_COLLISION"
        if self.stage == Stage.RECORDED_APPROACH:
            p_A = geo["center_A"]
            distance = float(np.linalg.norm(p_A))
            approaching = (
                self.previous_ring_distance is not None
                and distance < self.previous_ring_distance - 1.0e-6
            )
            self.previous_ring_distance = distance
            transition_radius = self.OUTER_RADIUS + self.pre_distance
            if distance <= transition_radius and approaching:
                self.stage = Stage.PREINSERTION_BLEND
                self.stage_time = 0.0
                self.blend_start_pose = (finger.center.copy(),
                                         quat_to_matrix(finger.quaternion_xyzw))
                pre_p, pre_r, _ = self._pose_target(ring, self.pre_distance)
                self.recorded_translation_correction = pre_p-finger.center
                self.recorded_orientation_correction_deg = math.degrees(
                    np.linalg.norm(rotation_vector(pre_r@self.blend_start_pose[1].T)))
                dp = np.linalg.norm(self.recorded_translation_correction)
                da = math.radians(self.recorded_orientation_correction_deg)
                # Minimum-jerk peak velocity is 1.875 delta/T.
                self.blend_duration = max(0.30, 1.875*dp/self.max_linear_velocity,
                                          1.875*da/self.max_angular_velocity)
            else:
                self._log(sim_time, frame, ring, geo, contact, None, False, 0, 0)
                return None, ring, geo
        self.stage_time += self.dt
        if self.stage == Stage.PREINSERTION_BLEND:
            p1, r1, p_A = self._pose_target(ring, self.pre_distance)
            p, r = self._smooth_pose(*self.blend_start_pose, p1, r1,
                                     self.stage_time/self.blend_duration)
            actual_position_error = float(np.linalg.norm(p1-finger.center))
            actual_orientation_error = float(np.linalg.norm(
                rotation_vector(r1@quat_to_matrix(finger.quaternion_xyzw).T)
            ))
            if self.stage_time >= self.blend_duration \
                    and actual_position_error < 0.002 \
                    and actual_orientation_error < math.radians(5.0):
                self.stage, self.stage_time = Stage.RING_RELATIVE_INSERTION, 0.0
        elif self.stage == Stage.RING_RELATIVE_INSERTION:
            duration = max(0.35, 1.875*(self.pre_distance+self.insertion_depth)
                           / self.max_linear_velocity)
            z = self.pre_distance + minimum_jerk(self.stage_time/duration) * (
                -self.insertion_depth-self.pre_distance)
            p, r, p_A = self._pose_target(ring, z)
            self.plane_crossed |= geo["crossed"]
            if self.stage_time >= duration and geo["inside"] and geo["crossed"] \
                    and geo["opposing_outside"]:
                self.stage, self.stage_time = Stage.PHYSICAL_GRASP, 0.0
                self.close_enabled = True
        elif self.stage == Stage.PHYSICAL_GRASP:
            p, r, p_A = self._pose_target(ring, -self.insertion_depth)
            self.close_enabled = True
            if contact.insertion_inner and contact.opposing_outer:
                self.grasp_hold += self.dt
            else:
                self.grasp_hold = 0.0
            self.bilateral |= self.grasp_hold >= 0.08
            if self.bilateral:
                self.stage, self.stage_time = Stage.PHYSICAL_PULL, 0.0
        elif self.stage == Stage.PHYSICAL_PULL:
            self.close_enabled = True
            pull_duration, pull_max = 0.8, 0.080
            self.pull_distance = pull_max*minimum_jerk(self.stage_time/pull_duration)
            p, r, p_A = self._pose_target(
                ring, -self.insertion_depth+self.pull_distance)
        else:
            self._log(sim_time, frame, ring, geo, contact, None, False, 0, 0)
            return None, ring, geo
        return (p, r, p_A), ring, geo

    def solve_dls(self, current_p, current_r, target_p, target_r, jacobian,
                  q, q_lower, q_upper, velocity_limit, acceleration_limit):
        self.target_count += 1
        ep = target_p-current_p
        er = rotation_vector(target_r@current_r.T)
        self.max_pos_error = max(self.max_pos_error, float(np.linalg.norm(ep)))
        self.max_rot_error = max(self.max_rot_error, float(np.linalg.norm(er)))
        error = np.r_[ep, er]
        j = np.asarray(jacobian, float)
        damping = 2e-3
        dq = 3.0*self.dt * (
            j.T @ np.linalg.solve(j@j.T+damping*damping*np.eye(6), error)
        )
        dq = np.clip(dq, -np.asarray(velocity_limit)*self.dt,
                     np.asarray(velocity_limit)*self.dt)
        if self.prev_dq is not None:
            delta = np.asarray(acceleration_limit)*self.dt*self.dt
            dq = np.clip(dq, self.prev_dq-delta, self.prev_dq+delta)
        candidate = np.clip(np.asarray(q)+dq, q_lower, q_upper)
        success = np.isfinite(candidate).all()
        if success:
            self.ik_success += 1
            self.max_joint_step = max(self.max_joint_step, float(np.max(np.abs(dq))))
            vel = dq/self.dt
            acc = (vel-self.prev_dq/self.dt)/self.dt if self.prev_dq is not None else vel/self.dt
            self.max_joint_velocity = max(self.max_joint_velocity, float(np.max(np.abs(vel))))
            self.max_joint_acceleration = max(self.max_joint_acceleration, float(np.max(np.abs(acc))))
            self.prev_dq = dq
            self.prev_q = candidate
        return candidate if success else None, float(np.linalg.norm(ep)), float(np.linalg.norm(er))

    def log_target(self, sim_time, frame, ring, geo, contact, target, success, pe, re, q):
        self._log(sim_time, frame, ring, geo, contact, target, success, pe, re)
        if success:
            self.q_log.append(np.asarray(q).copy())
            self.pose_log.append(np.r_[ring.local(target[0]), matrix_to_quat(ring.rotation.T@target[1])])
            self.stage_log.append(self.stage.value)

    def _log(self, sim_time, frame, ring, geo, contact, target, success, pe, re):
        self.frame_writer.writerow([sim_time,frame,self.stage.value,*ring.origin,
                                    *ring.rotation[:,0],*ring.rotation[:,1],*ring.rotation[:,2]])
        if target is not None:
            p_A = ring.local(target[0])
            q_A = matrix_to_quat(ring.rotation.T@target[1])
            self.target_writer.writerow([sim_time,frame,self.stage.value,*p_A,*q_A,
                                         geo["clearance"],int(success),pe,math.degrees(re)])
        self.contact_writer.writerow(
            [sim_time,frame,self.stage.value,int(geo["inside"]),int(geo["crossed"]),
             geo["clearance"],int(contact.insertion_inner),int(contact.opposing_outer),
             int(contact.side_collision),int(contact.phone_collision),
             contact.max_penetration_m,contact.pull_force_n,int(self.close_enabled)]
        )

    def note_detach(self, frame: int):
        self.detach_frame = frame
        self.stage = Stage.COMPLETE

    def close(self, left_hold: bool):
        for f in (self.frame_file,self.target_file,self.contact_file):
            f.flush(); f.close()
        np.savez(self.root/"ring_relative_right_arm_ik_trajectory.npz",
                 joint_position=np.asarray(self.q_log), target_pose_ring=np.asarray(self.pose_log),
                 stage=np.asarray(self.stage_log))
        ik_rate = self.ik_success/max(self.target_count, 1)
        insertion_pass = self.plane_crossed and not self.side_collision and self.minimum_clearance > .001
        grasp_pass = insertion_pass and self.bilateral and self.max_penetration < .001
        detach_pass = grasp_pass and self.detach_frame is not None and left_hold
        reports = {
            "ring_finger_full_orientation_report.txt": [
                "longitudinal_axis=-ring_Z", "wide_axis=ring_X", "thin_axis=ring_Y",
                f"target_rotation_world={None if self.target_rotation is None else self.target_rotation.tolist()}",
                "one_axis_alignment=False", "full_3d_orientation=True"],
            "ring_insertion_geometric_feasibility.txt": [
                f"projected_cross_section_m={self.CROSS_SECTION.tolist()}",
                f"minimum_clearance_m={self.minimum_clearance:.12g}",
                f"plane_crossed={self.plane_crossed}", f"side_collision={self.side_collision}",
                f"result={'RING_INSERTION_PASS' if insertion_pass else 'RING_INSERTION_FAIL'}"],
            "ring_preinsertion_blend_report.txt": [
                "blend_start_condition=distance<=outer_radius+geometry_derived_pre_distance AND approaching",
                f"pre_distance_m={self.pre_distance:.12g}",
                f"translation_correction_world_m={self.recorded_translation_correction.tolist()}",
                f"orientation_correction_deg={self.recorded_orientation_correction_deg:.12g}",
                f"blend_duration_s={self.blend_duration:.12g}", "interpolation=SE(3)_minimum_jerk"],
            "ring_relative_ik_report.txt": [
                f"total_target_count={self.target_count}", f"success_rate={ik_rate:.12g}",
                f"maximum_position_error_m={self.max_pos_error:.12g}",
                f"maximum_orientation_error_deg={math.degrees(self.max_rot_error):.12g}",
                f"maximum_joint_step_rad={self.max_joint_step:.12g}",
                f"maximum_joint_velocity_rad_s={self.max_joint_velocity:.12g}",
                f"maximum_joint_acceleration_rad_s2={self.max_joint_acceleration:.12g}"],
            "ring_bilateral_grasp_report.txt": [
                f"bilateral_contact={self.bilateral}", f"maximum_penetration_m={self.max_penetration:.12g}",
                f"result={'RING_GRASP_PASS' if grasp_pass else 'RING_GRASP_FAIL'}"],
            "ring_physical_pull_detach_report.txt": [
                f"maximum_outward_pull_force_n={self.max_pull_force:.12g}",
                f"detach_frame={self.detach_frame if self.detach_frame is not None else 'NONE'}",
                "frame_based_detach=False", f"left_phone_hold={left_hold}",
                f"result={'PHYSICAL_DETACH_PASS' if detach_pass else 'PHYSICAL_DETACH_FAIL'}"],
        }
        for name, lines in reports.items():
            with self._exclusive(name) as f:
                f.write("\n".join(lines)+"\n")
        return {"ik_rate":ik_rate, "insertion_pass":insertion_pass,
                "grasp_pass":grasp_pass, "detach_pass":detach_pass}
