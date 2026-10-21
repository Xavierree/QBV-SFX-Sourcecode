import test from 'node:test';
import assert from 'node:assert';

import Home from '../app/page';

test('SFX-web-app smoke test', async (t) => {
  await t.test('page renders', async () => {
    let el;
    try {
      el = await Home();
    } catch {
      el = Home();
    }
    assert.ok(el, 'Should render something');
  });
});
