declare class Keyv<Value = unknown> {
  constructor(options?: Keyv.Options<Value> | string);
  get<ExpectedValue = Value>(key: string): Promise<ExpectedValue | undefined>;
  set(key: string, value: Value, ttl?: number): Promise<boolean>;
  delete(key: string): Promise<boolean>;
  clear(): Promise<void>;
  has(key: string): Promise<boolean>;
  iterator(): AsyncIterableIterator<[string, Value]>;
}

declare namespace Keyv {
  interface Options<Value = unknown> {
    uri?: string;
    store?: unknown;
    namespace?: string;
    ttl?: number;
    serialize?: (data: Value) => string | Promise<string>;
    deserialize?: (data: string) => Value | Promise<Value>;
  }
}

export = Keyv;
