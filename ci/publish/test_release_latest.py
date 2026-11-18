import unittest
from unittest.mock import Mock, patch
import make_packages as driver
import publish_from_manifest as publisher

class LatestReleaseTests(unittest.TestCase):
    def release(self):
        return {'id':2,'assets':[{'name':f'{package}-26.10.0-py3-none-manylinux_2_34_{arch}.whl','state':'uploaded'}
            for package in publisher.SOC_TO_PACKAGE.values() for arch in publisher.ARCH_TO_WHEEL.values()]}

    def test_complete_newer_release_promoted(self):
        client=Mock(); client.find_latest_release.return_value={'id':1,'tag_name':'v26.9.1-release'}
        release=self.release()
        self.assertTrue(publisher.promote_complete_release(client,release,'26.10.0'))
        client.mark_latest.assert_called_once_with(release)

    def test_partial_or_older_release_not_promoted(self):
        client=Mock(); release=self.release(); release['assets'].pop()
        self.assertFalse(publisher.promote_complete_release(client,release,'26.10.0'))
        client.find_latest_release.assert_not_called()
        client.find_latest_release.return_value={'id':3,'tag_name':'v26.11.0-release'}
        self.assertFalse(publisher.promote_complete_release(client,self.release(),'26.10.0'))
        client.mark_latest.assert_not_called()

    def test_edit_preserves_latest_and_does_not_promote_old_release(self):
        client=driver.GitHubRelease('test/repo','fake','https://api.github.com',False)
        for current_id,expected in ((2,'true'),(3,'false')):
            with self.subTest(current_id=current_id), \
                 patch.object(client,'find_latest_release',return_value={'id':current_id}), \
                 patch.object(client,'_request') as request:
                client.update_body({'id':2},'body','title')
                self.assertEqual(request.call_args.kwargs['json_body']['make_latest'],expected)

if __name__=='__main__': unittest.main()
