import { describe, it, expect } from 'vitest';
import { humanizeError, isAuthError } from './api';

describe('humanizeError', () => {
  it('maps 401 to a re-auth message', () => {
    const e = humanizeError({ response: { status: 401, data: {} } });
    expect(e.kind).toBe('auth');
    expect(e.message).toMatch(/sign in/i);
  });

  it('maps 400 to the document-level input message', () => {
    const e = humanizeError({ response: { status: 400, data: { detail: 'scanned image' } } });
    expect(e.kind).toBe('input');
    expect(e.message).toBe('scanned image');
  });

  it('maps a network failure to a blameless system message', () => {
    const e = humanizeError({});
    expect(e.kind).toBe('network');
    expect(e.message).toMatch(/backend/i);
  });

  it('never leaks a raw stack trace', () => {
    const e = humanizeError({ response: { status: 500, data: {} } });
    expect(e.message).not.toMatch(/Traceback|at Object|\.py:/);
  });
});

describe('isAuthError', () => {
  it('detects 401 responses', () => {
    expect(isAuthError({ response: { status: 401 } })).toBe(true);
    expect(isAuthError({ response: { status: 500 } })).toBe(false);
    expect(isAuthError({})).toBe(false);
  });
});
