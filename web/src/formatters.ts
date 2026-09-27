/**
 * @file formatters.ts
 * Clinical and numerical formatting utilities using the ECMA-402 Internationalization API.
 */

import i18next from "i18next";

/**
 * Returns the currently active application locale identifier.
 * @returns {string} The active locale tag (e.g. 'en', 'ja', 'ar', 'he').
 */
export function getCurrentLocale(): string {
  if (i18next && i18next.language) {
    return i18next.language;
  }
  if (typeof document !== "undefined" && document.documentElement.lang) {
    return document.documentElement.lang;
  }
  return "en";
}

/**
 * Formats a numeric value according to the active or specified locale.
 * @param {number} value - The number to format.
 * @param {Intl.NumberFormatOptions} [options] - Optional formatting options.
 * @param {string} [locale] - Optional explicit BCP 47 language tag override.
 * @returns {string} The localized number string.
 */
export function formatNumber(
  value: number,
  options?: Intl.NumberFormatOptions,
  locale?: string,
): string {
  const targetLocale = locale || getCurrentLocale();
  return new Intl.NumberFormat(targetLocale, options).format(value);
}

/**
 * Formats a fractional or whole percentage according to the active locale.
 * @param {number} value - The percentage value (e.g. 70.5 for 70.5%).
 * @param {number} [decimals=1] - Maximum fractional digits.
 * @param {string} [locale] - Optional explicit BCP 47 language tag override.
 * @returns {string} The localized formatted percentage with symbol.
 */
export function formatPercent(
  value: number,
  decimals: number = 1,
  locale?: string,
): string {
  const targetLocale = locale || getCurrentLocale();
  const formattedVal = new Intl.NumberFormat(targetLocale, {
    minimumFractionDigits: 0,
    maximumFractionDigits: decimals,
  }).format(value);
  return `${formattedVal}%`;
}

/**
 * Formats a Date object or timestamp into a localized date string.
 * @param {Date | string | number} date - The date to format.
 * @param {Intl.DateTimeFormatOptions} [options] - Optional formatting options.
 * @param {string} [locale] - Optional explicit BCP 47 language tag override.
 * @returns {string} The localized date representation.
 */
export function formatDate(
  date: Date | string | number,
  options?: Intl.DateTimeFormatOptions,
  locale?: string,
): string {
  const targetLocale = locale || getCurrentLocale();
  const d = date instanceof Date ? date : new Date(date);
  return new Intl.DateTimeFormat(targetLocale, options).format(d);
}

/**
 * Formats a Date object or timestamp into a localized time string.
 * @param {Date | string | number} date - The date/time to format.
 * @param {Intl.DateTimeFormatOptions} [options] - Optional formatting options.
 * @param {string} [locale] - Optional explicit BCP 47 language tag override.
 * @returns {string} The localized time representation.
 */
export function formatTime(
  date: Date | string | number,
  options?: Intl.DateTimeFormatOptions,
  locale?: string,
): string {
  const targetLocale = locale || getCurrentLocale();
  const d = date instanceof Date ? date : new Date(date);
  const defaultOpts: Intl.DateTimeFormatOptions = {
    hour: "numeric",
    minute: "numeric",
    ...options,
  };
  return new Intl.DateTimeFormat(targetLocale, defaultOpts).format(d);
}
