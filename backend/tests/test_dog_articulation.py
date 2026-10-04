"""CPU algebra/contract tests; not an anatomical or image-quality acceptance."""
import unittest
import torch
from real_video.dog_articulation import build_dog_rig, DogRetargeter, PARENTS
from real_video.controller_rig import dual_quaternion_skin
from real_video.gaussian3d import forward_kinematics


def fixture():
    vertices = torch.tensor([[-.4946,-.00052179,-.2799791], [.5449,.7638,.3687],
                             [-.2,.5,.1], [.3,.4,-.1], [-.3,.1,.15], [.3,.1,.05]])
    faces = torch.tensor([[0,2,3], [1,2,3], [0,2,4], [1,3,5]])
    rig = build_dog_rig(vertices, faces)
    # An independent non-degenerate source skeleton in +Z forward coordinates.
    source = rig['joints']@rig['source_to_asset']
    source /= 1.7
    return vertices, faces, rig, source


class DogArticulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_weights_and_proper_basis(self):
        vertices, _, rig, _ = fixture()
        self.assertEqual(rig['vertex_weights'].shape, (len(vertices),27))
        self.assertTrue(bool((rig['vertex_weights']>=0).all()))
        torch.testing.assert_close(rig['vertex_weights'].sum(-1), torch.ones(len(vertices)))
        torch.testing.assert_close(torch.linalg.det(rig['source_to_asset']), torch.tensor(1.))

    def test_initial_guidance_exact_bind(self):
        vertices, _, rig, source = fixture()
        motion = DogRetargeter(rig,source)
        local, shift, joints, diagnostics = motion.step(source, torch.zeros(3))
        torch.testing.assert_close(local, torch.eye(3).repeat(27,1,1),atol=2e-6,rtol=1e-5)
        torch.testing.assert_close(shift, torch.zeros(3))
        torch.testing.assert_close(joints,rig['joints'])
        tf = forward_kinematics(rig['joints'],PARENTS,local)
        r = tf[:,:3,:3]
        offset = tf[:,:3,3]-(r@rig['joints'][...,None]).squeeze(-1)
        moved = dual_quaternion_skin(vertices,rig['vertex_weights'],r,offset)
        torch.testing.assert_close(moved,vertices,atol=2e-6,rtol=1e-5)
        self.assertLess(diagnostics['bone_length_max_error'],1e-6)

    def test_source_motion_changes_paws_with_fixed_bones(self):
        _,_,rig,source = fixture()
        modified = source.clone()
        modified[9,2] += .08
        modified[10,2] += .10
        local,_,joints,diag = DogRetargeter(rig,source).step(modified,torch.zeros(3))
        self.assertGreater(float((local[8]-torch.eye(3)).norm()), .1)
        self.assertGreater(float((joints[10]-rig['joints'][10]).norm()), .01)
        self.assertLess(diag['bone_length_max_error'],1e-6)
        torch.testing.assert_close(torch.linalg.det(local),torch.ones(27),atol=2e-5,rtol=1e-5)

    def test_geometry_scale_translation(self):
        vertices,faces,rig,_ = fixture()
        offset = torch.tensor([4.,-2.,1.])
        changed = build_dog_rig(vertices*2+offset,faces)
        torch.testing.assert_close(changed['joints'],rig['joints']*2+offset)
        torch.testing.assert_close(changed['vertex_weights'],rig['vertex_weights'],atol=2e-6,rtol=1e-5)

    def test_root_units_and_forward(self):
        _,_,rig,source = fixture()
        motion = DogRetargeter(rig,source)
        _,shift,_,_ = motion.step(source, torch.tensor([0.,0.,2.]))
        torch.testing.assert_close(shift, rig['source_to_asset'][:,2]*(2*motion.scale))
        self.assertLess(float(shift[0]),0)
        self.assertGreater(float(shift[2]),0)

    def test_no_mutation_and_repeat_determinism(self):
        _,_,rig,source = fixture()
        old=source.clone(); motion=DogRetargeter(rig,source)
        first=motion.step(source,torch.zeros(3))
        second=motion.step(source,torch.zeros(3))
        torch.testing.assert_close(source,old)
        for a,b in zip(first[:3],second[:3]):
            torch.testing.assert_close(a,b)

    def test_aligned_maps_target_bones_to_source_directions(self):
        _,_,rig,source = fixture()
        # Source need not share the target mesh bind offsets.
        source = source.clone()
        source[8] += torch.tensor([.01,.03,.08])
        motion=DogRetargeter(rig,source,mode='aligned')
        local,shift,joints,diag=motion.step(source,torch.zeros(3))
        for start,end in ((7,8),(8,9),(9,10),(16,17),(17,18),(18,19)):
            target_direction=torch.nn.functional.normalize(joints[end]-joints[start],dim=0)
            source_direction=torch.nn.functional.normalize((source[end]-source[start])@motion.basis.T,dim=0)
            torch.testing.assert_close(target_direction,source_direction,atol=2e-5,rtol=1e-5)
        self.assertLess(diag['bone_length_max_error'],1e-6)
        self.assertEqual(diag['retarget_mode'],'aligned')

    def test_paw_grounding_changes_only_root_height(self):
        _,_,rig,source=fixture()
        ungrounded=DogRetargeter(rig,source,mode='aligned').step(source,torch.tensor([0.,2.,0.]))
        grounded=DogRetargeter(rig,source,mode='aligned',ground_paws=True).step(source,torch.tensor([0.,2.,0.]))
        torch.testing.assert_close(ungrounded[0],grounded[0])
        torch.testing.assert_close(ungrounded[1][[0,2]],grounded[1][[0,2]])
        self.assertAlmostEqual(min(grounded[3]['paw_sole_y']),rig['floor'],places=5)
        self.assertLess(grounded[3]['root_grounding_delta_y'],-1.)

    def test_hybrid_preserves_relative_torso_and_aligns_only_limbs(self):
        _,_,rig,reference = fixture()
        source = reference.clone()
        source[8] += torch.tensor([.01,.03,.08])
        source[5] += torch.tensor([.02, .03, 0.])
        relative=DogRetargeter(rig,reference).step(source,torch.zeros(3))
        hybrid=DogRetargeter(rig,reference,mode='aligned_legs').step(source,torch.zeros(3))
        for index in (0,1,2,3,4,5,6,11,24,25,26):
            torch.testing.assert_close(relative[0][index],hybrid[0][index])
        for start,end in ((7,8),(8,9),(9,10),(16,17),(17,18),(18,19)):
            direction=torch.nn.functional.normalize(hybrid[2][end]-hybrid[2][start],dim=0)
            expected=torch.nn.functional.normalize((source[end]-source[start])@rig['source_to_asset'].T,dim=0)
            torch.testing.assert_close(direction,expected,atol=2e-5,rtol=1e-5)

    def test_reject_bad_pose_and_mesh(self):
        vertices,faces,rig,source=fixture(); motion=DogRetargeter(rig,source)
        with self.assertRaises(ValueError): motion.step(torch.zeros(26,3),torch.zeros(3))
        with self.assertRaises(ValueError): motion.step(source,torch.tensor([0.,float('nan'),0.]))
        with self.assertRaises(ValueError): build_dog_rig(vertices,faces+100)
        with self.assertRaises(ValueError): DogRetargeter(rig,source,scale=-1.)


if __name__ == '__main__': unittest.main()
